"""자율주행 주행 정책 (순수 로직, ROS/cv2/numpy 비의존 — 오프라인 단위 테스트 가능).

inference_node 가 YOLO 검출 + 차선 오프셋을 받아 /control(steering, throttle)을
산출하는 규칙을 담는다. CLAUDE.md 8.1/10.2 의 정책을 구현한다:
  - 출발 초록불 게이팅(B.1)      : require_green_start 확정 전 정지·중립.
  - 좌/우 갈림길 분기(B.3)       : left/right margin 게이팅 + turn_intent 래치 + 바이어스.
  - 도착 빨간불 하드 정지(B.6)    : 확정 시 throttle 0.0 으로 덮어씀.
  - 차선 추종 PD 조향             : offset PD + 슬루 제한 + 차선 로스트 시 마지막 조향 유지.
  - 커브 감속(B.2)               : |curvature| 홀드가 임계 이상이면 corner_throttle.
  - 순항 throttle                : 출발 후 cruise_throttle.
정/역 트랙은 drive_direction(±1)로 조향·분기 바이어스를 좌/우 미러링한다.

검출 결과(on_detections)는 YOLO 프레임 rate(≈3Hz)로, 조향/스로틀(step)은 제어
timer rate(≈20Hz)로 호출되는 것을 전제로 한다 — confirm_frames 는 YOLO 프레임,
fork_commit_frames 는 제어 프레임 단위.
"""
from dataclasses import dataclass

# 클래스 id (CLAUDE.md 10.1 정본)
REDLIGHT = 0
GREENLIGHT = 1
LEFT_SIGN = 2
RIGHT_SIGN = 3


@dataclass
class LaneSignal:
    """opencv_node 의 /lane/offset [offset, valid, curvature] 를 담는 값 객체."""
    offset: float = 0.0
    valid: bool = False
    curvature: float = 0.0


def default_params():
    """정책 파라미터 기본값(딕셔너리). inference_node 가 ROS param 으로 덮어씀.

    값 근거: CLAUDE.md 10.2 / auto_driving.launch.py.
    """
    return {
        'conf_threshold': 0.25,
        # 확정 프레임(YOLO 프레임)
        'start_confirm_frames': 2,   # 초록불 출발(B.1)
        'confirm_frames': 2,         # 좌/우 분기(B.3)
        'stop_confirm_frames': 2,    # 빨간불 정지(B.6)
        # 출발 게이트
        'require_green_start': True,
        # throttle
        'cruise_throttle': 0.13,
        'corner_throttle': 0.17,
        'turn_throttle': 0.13,
        'lane_lost_throttle': 0.10,
        # 커브 판정
        'corner_curvature_threshold': 0.30,
        'curve_hold_decay': 0.85,
        # 조향
        'steer_trim': 0.0,           # vehicle_config STEER_TRIM
        'steer_sign': -1.0,
        'steer_kp': 0.6,
        'steer_kd': 0.15,
        'steer_slew': 0.15,
        # 갈림길
        'turn_bias': 0.35,
        'fork_commit_frames': 30,    # 제어 프레임(30@20Hz≈1.5s)
        'sign_margin': 0.15,
        'sign_conf': 0.35,
        # 방향(정/역 트랙 미러링)
        'drive_direction': 1.0,
    }


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _slew(cur, target, max_step):
    delta = target - cur
    if delta > max_step:
        delta = max_step
    elif delta < -max_step:
        delta = -max_step
    return cur + delta


def resolve_sign(left_score, right_score, margin, conf):
    """한 프레임의 좌/우 표지판 점수로 방향을 결정(margin 게이팅).

    - 우세 점수가 conf 미만 → None(신뢰도 부족).
    - 좌우 점수 차가 margin 미만 → None(애매 → 대기, 오주행 방지).
    - 그 외 → 'left' | 'right'.
    좌/우 오분류가 곧 미션 실패이므로 분기류는 '미검출 시 대기'가 보수적(CLAUDE.md 9.5).
    """
    top = max(left_score, right_score)
    if top < conf:
        return None
    if abs(left_score - right_score) < margin:
        return None
    return 'left' if left_score > right_score else 'right'


class DrivingPolicy:
    """검출·차선 신호를 받아 (steering, throttle) 명령을 산출하는 상태 머신."""

    def __init__(self, params=None):
        p = default_params()
        if params:
            p.update(params)
        self.p = p

        # --- 래치 상태 ---
        # require_green_start=False 면 즉시 출발 상태로 시작(점검용).
        self.green_started = not p['require_green_start']
        self.red_stopped = False
        self.turn_intent = None       # 'left' | 'right' | None (첫 확정 래치)
        self.fork_remaining = 0

        # --- 연속 프레임 스트릭 ---
        self._green_streak = 0
        self._red_streak = 0
        self._left_streak = 0
        self._right_streak = 0

        # --- 제어 상태 ---
        self.last_steer = p['steer_trim']
        self.last_offset = 0.0
        self.corner_hold = 0.0

    # ---- YOLO 프레임 rate 로 호출 ----
    def on_detections(self, detections):
        """detections: class_id/score 속성을 가진 객체 리스트(YoloOnnx.Detection 등)."""
        p = self.p
        conf = p['conf_threshold']

        # 프레임 내 클래스별 최고 점수.
        best = {}
        for d in detections:
            if d.score >= conf and d.score > best.get(d.class_id, 0.0):
                best[d.class_id] = d.score

        # 초록/빨강 스트릭 → 래치.
        self._green_streak = self._green_streak + 1 if GREENLIGHT in best else 0
        self._red_streak = self._red_streak + 1 if REDLIGHT in best else 0
        if self._green_streak >= p['start_confirm_frames']:
            self.green_started = True
        if self._red_streak >= p['stop_confirm_frames']:
            self.red_stopped = True

        # 좌/우 margin 게이팅 후 스트릭 → turn_intent 래치(첫 확정 우선).
        side = resolve_sign(best.get(LEFT_SIGN, 0.0), best.get(RIGHT_SIGN, 0.0),
                            p['sign_margin'], p['sign_conf'])
        if side == 'left':
            self._left_streak += 1
            self._right_streak = 0
        elif side == 'right':
            self._right_streak += 1
            self._left_streak = 0
        else:
            self._left_streak = 0
            self._right_streak = 0

        if self.turn_intent is None:
            if self._left_streak >= p['confirm_frames']:
                self.turn_intent = 'left'
                self.fork_remaining = p['fork_commit_frames']
            elif self._right_streak >= p['confirm_frames']:
                self.turn_intent = 'right'
                self.fork_remaining = p['fork_commit_frames']

    # ---- 제어 timer rate 로 호출 ----
    def step(self, lane):
        """lane: LaneSignal. 반환 (steering, throttle), 둘 다 [-1,1]/[0,1] 범위."""
        p = self.p
        trim = p['steer_trim']

        # --- 조향: 차선 PD (유효 시) 또는 마지막 조향 유지(로스트 폴백) ---
        if lane.valid:
            offset = lane.offset
            d_off = offset - self.last_offset
            self.last_offset = offset
            effort = p['steer_sign'] * (p['steer_kp'] * offset + p['steer_kd'] * d_off)
            target = trim + effort
        else:
            target = self.last_steer

        # 갈림길 커밋 바이어스(정/역 미러링).
        committing = self.turn_intent is not None and self.fork_remaining > 0
        if committing:
            bias_dir = -1.0 if self.turn_intent == 'left' else 1.0
            target += p['steer_sign'] * p['drive_direction'] * bias_dir * p['turn_bias']
            self.fork_remaining -= 1

        target = _clamp(target, -1.0, 1.0)
        steer = _clamp(_slew(self.last_steer, target, p['steer_slew']), -1.0, 1.0)

        # --- 커브 감속 홀드(진입 전/중 감속 유지) ---
        curv = abs(lane.curvature) if lane.valid else 0.0
        self.corner_hold = max(self.corner_hold * p['curve_hold_decay'], curv)

        # --- throttle 우선순위: 출발게이트 > 빨간불 > 분기 > 커브 > 로스트 > 순항 ---
        if p['require_green_start'] and not self.green_started:
            steer = trim          # 출발 전 중립 유지
            throttle = 0.0
        elif self.red_stopped:
            throttle = 0.0        # 도착 빨간불 하드 정지
        elif committing:
            throttle = p['turn_throttle']
        elif self.corner_hold >= p['corner_curvature_threshold']:
            throttle = p['corner_throttle']
        elif not lane.valid:
            throttle = p['lane_lost_throttle']
        else:
            throttle = p['cruise_throttle']

        self.last_steer = steer
        return steer, throttle

    def state_summary(self):
        """디버그/로깅용 상태 스냅샷."""
        return {
            'green_started': self.green_started,
            'red_stopped': self.red_stopped,
            'turn_intent': self.turn_intent,
            'fork_remaining': self.fork_remaining,
            'corner_hold': round(self.corner_hold, 3),
        }
