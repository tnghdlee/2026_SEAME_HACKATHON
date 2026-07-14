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
        # 출발 대기 중 초록불 전용(더 낮은) 신뢰도 임계값 — 먼 신호등을 recall
        # 우선으로 잡는다(미인식=미션 실패, 오검출은 출발선에서 거의 무해).
        # green_started=False 일 때만 초록불에 적용, 그 외 클래스·출발 후에는
        # conf_threshold 를 쓴다. ⚠️ YOLO 출력 임계값(YoloOnnx.conf_threshold)도
        # 이 값까지 낮춰야 낮은 신뢰도 검출이 정책에 전달됨 — inference_node 참조.
        'green_start_conf': 0.12,
        # 확정 프레임(YOLO 프레임)
        'start_confirm_frames': 2,   # 초록불 출발(B.1)
        'confirm_frames': 2,         # 좌/우 분기(B.3)
        'stop_confirm_frames': 2,    # 빨간불 정지(B.6)
        # 출발 게이트
        'require_green_start': True,
        # 빨간불 오검출(ArUco 동적 장애물 구간의 '빨간 바닥') 배제 — 기하 게이팅.
        # 실제 신호등은 프레임 상단에 작게 잡히고, 빨간 바닥은 하단에 크게 잡힌다.
        # 이 두 값은 on_detections 에 frame_height 가 전달될 때만 적용된다(순수
        # 단위 테스트는 frame_height 미전달이라 필터 없이 종전대로 동작).
        # 박스 세로 중심이 프레임 높이의 이 비율보다 아래면 '바닥'으로 보고 무시.
        'redlight_max_y_ratio': 0.6,
        # 박스 높이가 프레임 높이의 이 비율 이상이면(=너무 큼) '바닥'으로 보고 무시.
        'redlight_max_h_ratio': 0.5,
        # 초록불 재확정 시 빨간불 하드 정지를 해제할지(정지/재출발 stop-go).
        # True: 빨간불로 멈춘 뒤 초록불을 다시 확정하면 재출발. 실코스에선 도착
        #   빨간불 뒤 초록불이 다시 나오지 않으므로 B.6 도착 영구정지와 무해하게
        #   양립하고, 정지/재출발 테스트도 가능하다.
        # False: 빨간불 정지를 영구 래치(엄격한 B.6 도착 종료 의미).
        'green_resumes_from_red': True,
        # 출발 킥스타트: 초록불 확정 직후 '조향 없이(중립) 고정 throttle 로 직진'
        # 하는 구간. 정지 상태에서 정지마찰을 이기고 곧게 출발시키기 위함. 구간
        # 동안 차선 PD·갈림길 분기·커브 감속을 모두 무시하고 steer=trim,
        # throttle=start_kick_throttle 을 낸다. 이후 정상 주행(차선 추종)으로 전환.
        # 프레임 단위는 제어 프레임(control_hz). inference_node 가 start_kick_seconds
        # ×control_hz 로 환산해 start_kick_frames 를 덮어쓴다(기본 2s@20Hz=40).
        'start_kick_throttle': 0.2,
        'start_kick_frames': 40,
        # throttle
        'cruise_throttle': 0.13,
        'corner_throttle': 0.17,
        'turn_throttle': 0.13,
        'lane_lost_throttle': 0.10,
        # 조향 감속: 조향 명령이 중립(trim)에서 steer_throttle_threshold 이상
        # 벗어나면(=바퀴를 꺾는 중) throttle 을 steer_throttle 로 낮춘다. 차선
        # 곡률 기반 corner_throttle 과 별개로, 실제 조향각에 직접 반응한다.
        'steer_throttle': 0.14,
        'steer_throttle_threshold': 0.05,
        # 커브 판정
        'corner_curvature_threshold': 0.30,
        'curve_hold_decay': 0.85,
        # 조향
        'steer_trim': 0.0,           # vehicle_config STEER_TRIM
        'steer_sign': -1.0,
        'steer_kp': 0.6,
        'steer_kd': 0.15,
        # 곡률 피드포워드(sim_line_260707_fix.py 에서 이식): 다가오는 커브의
        # 곡률에 비례해 조향을 '미리' 꺾어 커브 진입 이탈을 줄인다. offset PD 와
        # 같은 프레임(차선 기하)에서 나온 값이라 steer_sign 만 적용하고
        # drive_direction 미러링은 하지 않는다(turn_bias 와 다름). 직선 곡률
        # 노이즈(~0.04)엔 사실상 무영향, 실제 커브에서만 유효. 실차 튜닝 대상.
        'curve_ff': 0.30,
        'steer_slew': 0.15,
        # 갈림길
        'turn_bias': 0.5,            # 분기 방향 조향 바이어스(강하게 꺾어야 분기됨)
        'commit_lane_weight': 0.3,   # 커밋 중 차선 PD 기여 비중(0=차선 무시, 1=평소)
        'commit_steer_slew': 0.30,   # 커밋 중 조향 변화 상한(평소 steer_slew보다 큼)
        'fork_commit_frames': 30,    # 제어 프레임(30@20Hz≈1.5s)
        'sign_margin': 0.15,
        'sign_conf': 0.35,
        # 출발 직진 유예(제어 프레임): 초록불 출발 확정 후 이 프레임 수 동안은
        # 표지판 분기 래치를 억제해 직진(차선 추종)한다. 실코스가 출발→S자→
        # 갈림길 순서이므로 출발 직후 (오)검출된 표지판으로 즉시 꺾이지 않게 한다.
        # 유예가 끝나면 실제 갈림길에서 새로 confirm_frames 를 채워야 분기한다.
        # 0 이면 비활성(예전 동작: 표지판 확정 즉시 분기). 20Hz 기준 100≈5s.
        'start_straight_frames': 0,
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
        # 출발 직진 유예 잔여(제어 프레임). 출발 확정 전이에서 arm, step 에서 감소.
        self.start_straight_remaining = 0
        # 출발 킥스타트 잔여(제어 프레임). 초록불 확정 전이에서 arm, step 에서 감소.
        self.start_kick_remaining = 0

        # --- 연속 프레임 스트릭 ---
        self._green_streak = 0
        self._red_streak = 0
        self._left_streak = 0
        self._right_streak = 0

        # --- 제어 상태 ---
        self.last_steer = p['steer_trim']
        self.last_offset = 0.0
        self.corner_hold = 0.0

    def _redlight_is_real(self, det, frame_height):
        """빨간불 검출이 실제 신호등인지(바닥 오검출이 아닌지) 기하로 판정.

        ArUco 동적 장애물 구간(B.4)의 '빨간 바닥'이 redlight 로 오검출돼 차가
        잘못 정지하는 문제를 막는다. 실제 신호등은 카메라 프레임 상단에 작게
        잡히고, 빨간 바닥은 하단에 크게 잡히므로 박스의 위치(세로 중심)와
        크기(높이)로 바닥을 배제한다. frame_height 미전달(순수 단위 테스트 등)
        시엔 기하 정보가 없으므로 필터를 적용하지 않는다(종전 동작 유지).
        """
        if not frame_height:
            return True
        p = self.p
        h = float(frame_height)
        cy = 0.5 * (det.y1 + det.y2)          # 박스 세로 중심
        box_h = det.y2 - det.y1
        if cy > p['redlight_max_y_ratio'] * h:
            return False                       # 하단 → 바닥으로 간주
        if box_h > p['redlight_max_h_ratio'] * h:
            return False                       # 지나치게 큰 박스 → 바닥으로 간주
        return True

    # ---- YOLO 프레임 rate 로 호출 ----
    def on_detections(self, detections, frame_height=None, frame_width=None):
        """detections: class_id/score/x1..y2 속성을 가진 객체 리스트(YoloOnnx.Detection 등).

        frame_height 를 넘기면 빨간불 오검출(빨간 바닥, B.4) 기하 게이팅이
        활성화된다(_redlight_is_real). frame_width 는 향후 확장용(현재 미사용).
        """
        p = self.p
        conf = p['conf_threshold']

        # 프레임 내 클래스별 최고 점수. 클래스별 신뢰도 임계값을 적용한다:
        #  - 출발 전 초록불: green_start_conf(완화) → 먼 신호등 recall 우선.
        #  - 그 외/출발 후: conf_threshold.
        # 빨간불은 바닥 오검출을 기하로 배제(_redlight_is_real).
        best = {}
        for d in detections:
            cls = d.class_id
            if cls == GREENLIGHT and not self.green_started:
                thr = p['green_start_conf']
            else:
                thr = conf
            if d.score < thr:
                continue
            if cls == REDLIGHT and not self._redlight_is_real(d, frame_height):
                continue
            if d.score > best.get(cls, 0.0):
                best[cls] = d.score

        # 초록/빨강 스트릭 → 래치.
        self._green_streak = self._green_streak + 1 if GREENLIGHT in best else 0
        self._red_streak = self._red_streak + 1 if REDLIGHT in best else 0
        if self._green_streak >= p['start_confirm_frames']:
            # 출발 순간(최초 확정 전이): 조향을 중앙으로 정렬한다. 출발 전에
            # 잘못 래치된 분기 의도(turn_intent)와 조향 상태를 리셋해, 초록불로
            # 출발하자마자 스테일 커밋으로 꺾이지 않고 직진/차선중앙에서 시작한다.
            # 출발 이후 실제로 표지판을 보면 다시 정상적으로 래치된다.
            if not self.green_started:
                self.turn_intent = None
                self.fork_remaining = 0
                self._left_streak = 0
                self._right_streak = 0
                self.last_steer = p['steer_trim']
                self.last_offset = 0.0
                self.corner_hold = 0.0
                # 출발 직후 직진 유예 arm — 이 구간 동안 분기 래치 억제.
                self.start_straight_remaining = p['start_straight_frames']
                # 출발 킥스타트 arm — 조향 없이 고정 throttle 로 직진 출발.
                self.start_kick_remaining = p['start_kick_frames']
            self.green_started = True
            # 초록불 재확정 시 빨간불 정지 해제(정지/재출발). 도착 영구정지를
            # 원하면 green_resumes_from_red=False 로 끈다.
            if p['green_resumes_from_red']:
                self.red_stopped = False
                self._red_streak = 0
        if self._red_streak >= p['stop_confirm_frames']:
            self.red_stopped = True

        # 좌/우 margin 게이팅 후 스트릭 → turn_intent 래치(첫 확정 우선).
        side = resolve_sign(best.get(LEFT_SIGN, 0.0), best.get(RIGHT_SIGN, 0.0),
                            p['sign_margin'], p['sign_conf'])
        # 출발 직진 유예 중에는 표지판을 무시(직진 유지). side=None 으로 만들면
        # 아래 else 가지에서 좌/우 스트릭이 0 으로 리셋되어, 유예 종료 후 실제
        # 갈림길에서 새로 confirm_frames 를 채워야 분기한다.
        if self.green_started and self.start_straight_remaining > 0:
            side = None
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
        # trim 기준 상대량(effort)만 계산해 두고, 커밋 여부에 따라 차선 기여도를
        # 조절한다. lane_effort = (차선추종 목표 - trim).
        if lane.valid:
            offset = lane.offset
            d_off = offset - self.last_offset
            self.last_offset = offset
            # PD(현재 오차) + 곡률 피드포워드(다가오는 커브 예측). curvature 는
            # offset 과 같은 부호 규약(먼 밴드가 오른쪽으로 휘면 +)이라 steer_sign
            # 만 곱한다 — drive_direction 미러링 대상 아님(offset PD 와 동일).
            lane_effort = p['steer_sign'] * (
                p['steer_kp'] * offset
                + p['steer_kd'] * d_off
                + p['curve_ff'] * lane.curvature)
        else:
            lane_effort = self.last_steer - trim  # 로스트 시 마지막 조향 유지

        # 갈림길 커밋: 분기 바이어스가 지배하도록 차선 PD 기여를 약화한다.
        # 갈림길에선 두 갈래가 모두 보여 차선 신호가 애매하므로, 커밋 중에는
        # 차선 추종을 commit_lane_weight(0~1)로 낮추고 turn_bias 로 분기 방향을
        # 강하게 밀어야 실제로 꺾인다. (예전엔 PD가 바이어스를 상쇄해 조향이
        # 안 됐다 — 속도만 줄고 방향 전환 실패.)
        # 출발 게이트 중(초록불 확정 전)에는 조향이 trim 으로 강제되므로 커밋을
        # 소진하지 않는다 — 출발 전 표지판이 보여도 fork_remaining 이 낭비되지
        # 않게 해, 실제 출발 후 갈림길에서 온전한 커밋 창을 쓴다.
        gated = p['require_green_start'] and not self.green_started
        # 출발 후 직진 유예 카운트다운(출발 확정·게이트 해제 후에만 감소).
        if self.green_started and not gated and self.start_straight_remaining > 0:
            self.start_straight_remaining -= 1
        # 출발 킥스타트 구간: 초록불 출발 후(빨간불 아님) 잔여가 남아 있으면,
        # 조향 없이(trim) 고정 throttle 로 직진 출발한다(정지마찰 극복). 빨간불이면
        # 잔여를 소진하지 않고 대기(정지 우선).
        kicking = (self.green_started and not gated
                   and not self.red_stopped
                   and self.start_kick_remaining > 0)
        committing = (self.turn_intent is not None and self.fork_remaining > 0
                      and not gated)
        if committing:
            bias_dir = -1.0 if self.turn_intent == 'left' else 1.0
            bias = p['steer_sign'] * p['drive_direction'] * bias_dir * p['turn_bias']
            target = trim + p['commit_lane_weight'] * lane_effort + bias
            self.fork_remaining -= 1
        else:
            target = trim + lane_effort

        target = _clamp(target, -1.0, 1.0)
        # 커밋 중엔 분기를 신속히 완성하도록 슬루 상한을 완화(commit_steer_slew).
        slew_limit = p['commit_steer_slew'] if committing else p['steer_slew']
        steer = _clamp(_slew(self.last_steer, target, slew_limit), -1.0, 1.0)

        # --- 커브 감속 홀드(진입 전/중 감속 유지) ---
        curv = abs(lane.curvature) if lane.valid else 0.0
        self.corner_hold = max(self.corner_hold * p['curve_hold_decay'], curv)

        # --- throttle 우선순위: 출발게이트 > 빨간불 > 킥스타트 > 분기 > 로스트 > 조향 > 커브 > 순항 ---
        if p['require_green_start'] and not self.green_started:
            steer = trim          # 출발 전 중립 유지
            throttle = 0.0
        elif self.red_stopped:
            throttle = 0.0        # 도착 빨간불 하드 정지
        elif kicking:
            steer = trim          # 킥스타트: 조향 없이 직진
            throttle = p['start_kick_throttle']
            self.start_kick_remaining -= 1
        elif committing:
            throttle = p['turn_throttle']
        elif not lane.valid:
            throttle = p['lane_lost_throttle']
        elif abs(steer - trim) >= p['steer_throttle_threshold']:
            throttle = p['steer_throttle']   # 조향 중(바퀴 꺾는 중) 감속
        elif self.corner_hold >= p['corner_curvature_threshold']:
            throttle = p['corner_throttle']
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
            'start_kick_remaining': self.start_kick_remaining,
        }
