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
        # 빨강 소멸 출발(대안 트리거): 대기 중 켜진 빨간불이 '확정 정지(red_stopped)'
        # 될 만큼 확실히 잡힌 뒤, 빨강이 red_gone_frames(YOLO 프레임) 연속 사라지면
        # = 초록 점등으로 보고 출발한다. 초록 검출이 약해도(빨강은 강함) 출발을
        # 놓치지 않기 위한 것으로, 기존 초록 확정 출발과 병행(둘 중 뭐라도 출발).
        # 전제: 대기 중 신호등이 빨강을 표시(사용자 확인). red_gone_frames 는 빨강
        # 검출이 간헐적으로 깜빡여도 오출발하지 않도록 실차 튜닝(크면 반응 느림).
        'start_on_red_gone': True,
        'red_gone_frames': 5,
        # 신호등(빨강/초록) 공통 ROI 게이팅: 신호등은 트랙 위쪽에 설치되므로 검출
        # 박스의 세로 중심이 프레임 높이의 이 비율보다 아래면(=하단) 오검출로 보고
        # 무시한다. 빨강·초록 둘 다에 적용(상단 50% 로 제한). frame_height 가
        # on_detections 에 전달될 때만 적용(순수 단위 테스트는 미전달 → 종전 동작).
        'light_roi_top_ratio': 0.5,
        # 신호등(빨강/초록) 공통 가로 ROI 게이팅: 신호등이 화면의 특정 가로 영역에만
        # 보일 때(예: 좌상단 설치), 그 밖(반대쪽·중앙)의 오검출을 배제한다. 박스
        # 세로 중심의 가로 위치(cx)가 프레임 폭의 [x_min, x_max] 안일 때만 유효.
        # 기본값 0.0~1.0=전체 폭(가로 제한 없음, 종전 동작). frame_width 가
        # on_detections 에 전달될 때만 적용(순수 단위 테스트는 미전달 → 종전 동작).
        'light_roi_x_min': 0.0,
        'light_roi_x_max': 1.0,
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
        'start_kick_throttle': 0.18,
        'start_kick_frames': 40,
        # throttle
        'cruise_throttle': 0.18,
        'corner_throttle': 0.17,
        'turn_throttle': 0.18,
        'lane_lost_throttle': 0.18,
        # 조향 감속: 조향 명령이 중립(trim)에서 steer_throttle_threshold 이상
        # 벗어나면(=바퀴를 꺾는 중) throttle 을 steer_throttle 로 낮춘다. 차선
        # 곡률 기반 corner_throttle 과 별개로, 실제 조향각에 직접 반응한다.
        'steer_throttle': 0.17,
        'steer_throttle_threshold': 0.05,
        # 커브 판정
        'corner_curvature_threshold': 0.30,
        'curve_hold_decay': 0.85,
        # 조향
        'steer_trim': 0.0,           # vehicle_config STEER_TRIM
        # 트림 기준 대칭 조향 클램프. 서보는 명령 [-1,1] 을 기하학적 중심(1500µs)
        # 기준 대칭으로 매핑하는데(d3racer.set_steering_percent), 정책 중립은
        # steer_trim(≠0)이라 두 중심이 어긋난다. 그 결과 도달 가능한 최대 조향각이
        # 좌우 비대칭이 된다: 한쪽 |1-trim|, 반대쪽 |1+trim|(trim=0.1 → 0.9 vs 1.1,
        # ≈450µs vs 550µs). 여유 적은 쪽은 최대각 미달, 큰 쪽은 코너에서 과조향.
        # True 면 조향 명령을 [trim-half, trim+half](half=1-|trim|)로 대칭 클램프해
        # 강한 쪽을 약한 쪽에 맞춘다(과조향 제거). trim=0 이면 [-1,1] 로 무동작.
        # ⚠️ 이는 SW 완화책이다 — 약한 쪽의 최대각을 늘리려면 서보 혼/링키지를
        # 기계적으로 재중심화해 STEER_TRIM 을 0 근처로 낮춰야 양쪽 ±500µs 회복.
        'symmetric_steer': True,
        'steer_sign': -1.0,
        'steer_kp': 0.4,
        'steer_kd': 0.5,
        # 조향 데드밴드: |offset|<이 값이면 비례항 0(직선 지그재그/hunting 방지).
        'steer_deadband': 0.04,
        # 곡률 피드포워드(sim_line_260707_fix.py 에서 이식): 다가오는 커브의
        # 곡률에 비례해 조향을 '미리' 꺾어 커브 진입 이탈을 줄인다. offset PD 와
        # 같은 프레임(차선 기하)에서 나온 값이라 steer_sign 만 적용하고
        # drive_direction 미러링은 하지 않는다(turn_bias 와 다름). 직선 곡률
        # 노이즈(~0.04)엔 사실상 무영향, 실제 커브에서만 유효. 실차 튜닝 대상.
        'curve_ff': 0.35,
        'steer_slew': 0.15,
        # 갈림길
        'turn_bias': 0.5,            # 분기 방향 조향 바이어스(강하게 꺾어야 분기됨)
        'commit_lane_weight': 0.3,   # 커밋 중 차선 PD 기여 비중(0=차선 무시, 1=평소)
        'commit_steer_slew': 0.30,   # 커밋 중 조향 변화 상한(평소 steer_slew보다 큼)
        'fork_commit_frames': 30,    # 제어 프레임(30@20Hz≈1.5s)
        'sign_margin': 0.15,
        'sign_conf': 0.35,
        # --- 방향 표지판 위치·크기(ROI) 게이팅 (B.3, 신설) ---
        # 신호등엔 _light_in_roi/_redlight_is_real 기하 게이팅이 있으나 방향 표지판은
        # 점수(margin/conf)만 봤다. 그 결과 모델이 배경/트랙을 표지판으로 오인해 뱉는
        # '화면 대부분을 덮는 거대 박스'(실측: w~프레임폭, h~프레임높이)가 그대로
        # turn_intent 를 래치했다. 여기에 박스 중심 ROI + 면적비 상·하한을 걸어
        # 비현실적 검출을 배제한다. 실제 표지판은 '중앙/정면'에 보이므로(사용자 확인)
        # 좌/우는 위치로 구분되지 않고 클래스 라벨에 의존한다 → ROI 는 좌/우 구분이
        # 아니라 '거대 오검출 배제' 용도.
        # ⚠️ 이 게이팅은 on_detections 에 frame_height/width 가 전달될 때만 활성이다.
        # 미전달(순수 단위 테스트)시엔 종전 동작(게이팅 없음)을 유지한다
        # (_light_in_roi 등과 동일한 하위호환 규약).
        # 기본값은 '사실상 비활성'(전체화면 ROI + 면적 0~1) — 녹화 캘리브레이션 전까지
        # 기존 동작을 바꾸지 않는다. 녹화 분석(tools/analyze_sign_roi.py) 후 실측
        # 값으로 좁힌다.
        # 박스 중심(cx,cy)/프레임 정규화 좌표가 이 사각형[x0,y0,x1,y1] 안일 때만 인정.
        'sign_roi_norm': (0.0, 0.0, 1.0, 1.0),
        # 박스 면적/프레임 면적이 이 하한 미만이면 무시(노이즈성 소형 오검출).
        'sign_min_area_ratio': 0.0,
        # 박스 면적/프레임 면적이 이 상한 초과면 무시(화면 대부분을 덮는 거대 오검출).
        # 1.0=상한 비활성. 캘리브레이션 후 예: 0.6 으로 낮춰 거대 박스를 배제.
        'sign_max_area_ratio': 1.0,
        # 커밋(꺾기) 전 방향 래치 갱신: 이미 래치된 turn_intent 와 '반대' 방향이 이
        # 프레임 수만큼 연속 확정되면 래치를 갱신한다. 출발 직후 (오)검출로 잘못
        # 래치된 방향을 실제 갈림길 표지판이 교정하게 해, "먼저 잡힌 방향이 영구
        # 고정 → 오주행"(B.3 미션 실패)을 막는다. 초기 래치(confirm_frames=2)보다
        # 엄격한 증거를 요구하도록 크게 잡는다. 커밋 이후엔 무시(방향 고정, 안전).
        'sign_revise_frames': 3,
        # --- 갈림길 커밋 트리거(근접/소실, B.3) ---
        # 방향(turn_intent)은 표지판을 멀리서 봐도 confirm_frames 로 일찍 래치
        # (기억)하되, 실제로 꺾는 커밋은 표지판이 충분히 '가까워졌을 때'만 시작
        # 한다. 멀리서 확정하자마자 꺾어 코스를 벗어나는 것을 막는다(사용자 요구:
        # "멀리서 봐도 참았다가 갈림길에서 차선 따라 좌/우로 나눠 간다").
        # ⚠️ 이 게이팅은 on_detections 에 frame_height 가 전달될 때만 활성이다.
        # 미전달(순수 단위 테스트)시엔 기하 정보가 없어 종전 동작(래치 즉시 커밋)
        # 을 유지한다(_light_in_roi/_redlight_is_real 와 동일한 하위호환 규약).
        #
        # 근접 지표(sign_proximity_metric) — 카메라 지오메트리에 맞게 선택:
        #   'bottom_y' : 표지판 박스 하단(y2)의 세로 위치/프레임 높이. 가까워질수록
        #                프레임 아래로 내려감. 카메라가 높아 내려다보고 표지판이
        #                낮게(지면 근처) 설치된 경우 가장 견고(단조·큰 레인지). [기본]
        #   'area'     : 박스 면적/프레임 면적. 표지판이 카메라 높이쯤 떠 있어
        #                세로 위치가 안 변할 때. 폭은 원근눌림이 덜해 높이보다 나음.
        #   'height'   : 박스 높이/프레임 높이. 카메라가 낮고 표지판을 정면으로 볼 때.
        #                카메라가 높으면 근접 시 높이가 포화·눌림(비단조)이라 비권장.
        # 지표값(0~1) ≥ sign_commit_ratio 이면 '가까움'으로 보고 커밋 시작.
        # ⚠️ 실트랙 캘리브레이션 필수 — 지표를 바꾸면 임계값도 다시 잡아야 한다
        #    (bottom_y≈0.55~0.7, area≈0.03~0.10, height≈0.15~0.30 대략).
        'sign_proximity_metric': 'bottom_y',
        'sign_commit_ratio': 0.60,
        # 소실 폴백(백업): 표지판이 프레임 밖으로 벗어나 사라지는 경우 대비. 래치된
        # 방향의 표지판 근접지표가 sign_lost_min_ratio 이상으로 커진 적 있고(=가까이
        # 왔었고), 이후 sign_lost_commit_frames(YOLO 프레임) 연속 미검출이면 커밋한다.
        # 표지판이 프레임 안에 계속 보이면 근접 임계가 먼저 걸려 이 백업은 안 쓰인다.
        'sign_lost_min_ratio': 0.45,
        'sign_lost_commit_frames': 3,
        # 백스톱 타임아웃(YOLO 프레임): 방향을 래치한 뒤 '검출된' 표지판을 누적
        # 이 프레임 수 이상 봤는데도 근접(sign_commit_ratio)/소실 트리거가 아직
        # 커밋을 못 걸었으면(근접 임계 미달·검출 깜빡임 등 캘리브레이션 어긋남),
        # 그대로 직진해 표지판을 들이받지 않도록 강제로 커밋한다. 누적 '검출'
        # 프레임만 세므로 멀리서 잠깐 잡혔다 사라진 표지판(_sign_far_blip)에는
        # 발동하지 않는다. 0 이면 비활성(종전 동작).
        # ⚠️ 근본 해결은 sign_commit_ratio 캘리브레이션(tools/sign_commit_calibration.py).
        #    이 값은 안전망이라 너무 작으면 갈림길 도달 전에 조기 커밋할 수 있다.
        'sign_commit_timeout_frames': 10,
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


def _deadband(x, dead):
    """중앙(0) 근처 |x|<dead 를 0 으로, 바깥은 dead 만큼 당겨 연속 유지(소프트 데드밴드).

    직선에서 차선 offset 노이즈로 조향이 좌우로 떠는(hunting/지그재그) 것을 막는다.
    임계에서 튐이 없도록 threshold 를 빼서 연속으로 이어붙인다.
    """
    if dead <= 0.0:
        return x
    if x > dead:
        return x - dead
    if x < -dead:
        return x + dead
    return 0.0


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
        # 갈림길 커밋 트리거 상태: 방향은 멀리서 일찍 래치하되, 실제 꺾기
        # (fork_remaining>0)는 표지판이 가까워졌을 때만 시작한다. commit_triggered
        # 는 한 번 커밋을 시작하면 다시 무장하지 않게 하는 래치(1회 분기).
        self.commit_triggered = False
        self._sign_max_prox = 0.0      # 래치 후 표지판 최대 근접지표(소실 폴백용)
        self._sign_absent_streak = 0   # 래치 후 표지판 연속 미검출(YOLO 프레임)
        self._sign_seen_since_latch = 0  # 래치 후 표지판을 본 누적 YOLO 프레임(백스톱)
        # 출발 직진 유예 잔여(제어 프레임). 출발 확정 전이에서 arm, step 에서 감소.
        self.start_straight_remaining = 0
        # 출발 킥스타트 잔여(제어 프레임). 초록불 확정 전이에서 arm, step 에서 감소.
        self.start_kick_remaining = 0

        # --- 연속 프레임 스트릭 ---
        self._green_streak = 0
        self._red_streak = 0
        self._red_gone_streak = 0     # red_stopped 무장 후 빨강 연속 미검출(빨강 소멸 출발)
        self._left_streak = 0
        self._right_streak = 0

        # --- 제어 상태 ---
        self.last_steer = p['steer_trim']
        self.last_offset = 0.0
        self.corner_hold = 0.0

        # 트림 기준 대칭 조향 한계 사전 계산(step 에서 명령 클램프에 사용).
        # half = 1-|trim| → lo=trim-half, hi=trim+half. trim=0 → [-1,1](무동작).
        trim = p['steer_trim']
        if p.get('symmetric_steer', True):
            half = max(0.0, 1.0 - abs(trim))
            self._steer_lo = trim - half
            self._steer_hi = trim + half
        else:
            self._steer_lo, self._steer_hi = -1.0, 1.0

    def _light_in_roi(self, det, frame_height, frame_width=None):
        """신호등(빨강/초록) 검출을 프레임 상단(+선택적 가로) ROI 로 제한.

        신호등은 트랙 위쪽에 설치되므로, 박스 세로 중심이 프레임 상단
        light_roi_top_ratio(기본 0.5=상단 50%) 안에 있을 때만 유효로 본다.
        하단 검출은 바닥 반사·배경 오검출로 배제한다. 추가로 frame_width 가
        전달되면 박스 가로 중심(cx)이 [light_roi_x_min, light_roi_x_max]×폭 안일
        때만 유효로 봐, 신호등이 좌/우 특정 영역에만 보일 때 반대쪽 오검출을
        배제한다(기본 0~1=가로 제한 없음). frame_height/width 미전달(순수 단위
        테스트)시엔 해당 축 필터 미적용(종전 동작 유지).
        """
        if not frame_height:
            return True
        cy = 0.5 * (det.y1 + det.y2)          # 박스 세로 중심
        if cy > self.p['light_roi_top_ratio'] * float(frame_height):
            return False
        if frame_width:
            cx = 0.5 * (det.x1 + det.x2)      # 박스 가로 중심
            w = float(frame_width)
            if (cx < self.p['light_roi_x_min'] * w
                    or cx > self.p['light_roi_x_max'] * w):
                return False
        return True

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

    def _sign_is_real(self, det, frame_height, frame_width):
        """방향 표지판 검출이 현실적인지(거대 오검출·노이즈가 아닌지) 기하로 판정.

        모델이 배경/트랙을 표지판으로 오인해 뱉는 '화면 대부분을 덮는 거대 박스'를
        박스 면적비(상·하한)와 중심 ROI 로 배제한다. frame_height/width 미전달(순수
        단위 테스트)시엔 기하 정보가 없어 필터 미적용(종전 동작 유지, 하위호환).
        """
        if not frame_height or not frame_width:
            return True
        p = self.p
        h = float(frame_height)
        w = float(frame_width)
        box_w = det.x2 - det.x1
        box_h = det.y2 - det.y1
        if box_w <= 0.0 or box_h <= 0.0:
            return False
        area_ratio = (box_w * box_h) / (w * h)
        if area_ratio < p['sign_min_area_ratio']:
            return False                       # 너무 작음 → 노이즈
        if area_ratio > p['sign_max_area_ratio']:
            return False                       # 너무 큼 → 거대 오검출
        roi = p['sign_roi_norm']
        if roi and len(roi) == 4:
            cx = 0.5 * (det.x1 + det.x2) / w   # 박스 중심(정규화)
            cy = 0.5 * (det.y1 + det.y2) / h
            x0, y0, x1, y1 = roi
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                return False                   # ROI 밖 → 무시
        return True

    # ---- 갈림길 커밋 트리거(근접/소실) ----
    def _sign_proximity(self, det, frame_height, frame_width):
        """표지판 박스의 '근접 정도'를 0~1 스칼라로 환산(sign_proximity_metric).

        가까울수록 값이 커지도록 정의한다:
          'bottom_y' : 박스 하단(y2)의 세로 위치/프레임 높이. 카메라가 높아
                       내려다볼 때 가까울수록 표지판이 프레임 아래로 내려감(단조).
          'area'     : 박스 면적/프레임 면적. frame_width 없으면 높이²로 폴백.
          'height'   : 박스 높이/프레임 높이.
        """
        h = float(frame_height)
        metric = self.p['sign_proximity_metric']
        if metric == 'height':
            return (det.y2 - det.y1) / h
        if metric == 'area':
            w = float(frame_width) if frame_width else h
            return ((det.x2 - det.x1) * (det.y2 - det.y1)) / (w * h)
        # 기본 'bottom_y'(미지의 metric 문자열도 여기로 폴백).
        return det.y2 / h

    def _maybe_trigger_commit(self, best_box, frame_height, frame_width):
        """래치된 방향의 표지판이 충분히 가까워졌는지(또는 프레임 밖으로 벗어나
        사라졌는지) 판단해 커밋(fork_remaining)을 시작한다. turn_intent 가 이미
        래치됐고 아직 커밋 전(commit_triggered=False)일 때만 호출된다.

        - frame_height 미전달(순수 단위 테스트): 기하 정보가 없으므로 종전 동작
          (래치 즉시 커밋). 하위호환.
        - 근접: 래치 방향 표지판의 근접지표(_sign_proximity) ≥ sign_commit_ratio
          → 커밋. 표지판이 가까워질수록 지표가 단조 증가하는 성질을 이용.
        - 소실 폴백(백업): 지표가 sign_lost_min_ratio 이상으로 커진 적 있고(가까이
          왔었고) 이후 sign_lost_commit_frames(YOLO 프레임) 연속 미검출이면,
          표지판이 프레임 밖으로 벗어난 것으로 보고 커밋.
        - 백스톱 타임아웃: 래치 후 표지판을 '검출된' 상태로 누적
          sign_commit_timeout_frames 프레임 이상 봤는데도 위 둘이 커밋을 못 걸었으면
          (근접 임계 미달·검출 깜빡임 등 캘리브레이션 어긋남), 들이받기 전에 강제
          커밋한다. 누적 '검출' 프레임만 세므로 멀리서 잠깐 잡혔다 사라진 표지판엔
          발동하지 않는다(sign_commit_timeout_frames=0 이면 비활성).
        """
        p = self.p
        if not frame_height:
            self._start_commit()               # 하위호환: 기하 없으면 즉시 커밋
            return
        sign_cls = LEFT_SIGN if self.turn_intent == 'left' else RIGHT_SIGN
        det = best_box.get(sign_cls)
        if det is not None:
            self._sign_seen_since_latch += 1
            prox = self._sign_proximity(det, frame_height, frame_width)
            if prox > self._sign_max_prox:
                self._sign_max_prox = prox
            self._sign_absent_streak = 0
            if prox >= p['sign_commit_ratio']:
                self._start_commit()           # 근접 → 커밋
                return
        else:
            self._sign_absent_streak += 1
            if (self._sign_max_prox >= p['sign_lost_min_ratio']
                    and self._sign_absent_streak >= p['sign_lost_commit_frames']):
                self._start_commit()           # 소실 폴백 → 커밋
                return
        # 백스톱: 근접·소실이 아직 커밋을 못 걸었어도, 표지판을 충분히 오래
        # 본(누적) 뒤엔 강제로 커밋해 직진 충돌을 막는다.
        timeout = p['sign_commit_timeout_frames']
        if timeout and self._sign_seen_since_latch >= timeout:
            self._start_commit()               # 백스톱 타임아웃 → 커밋

    def _reset_sign_tracking(self):
        """방향 래치 갱신 시 근접/소실 추적을 초기화(새 방향으로 커밋 재판정)."""
        self._sign_max_prox = 0.0
        self._sign_absent_streak = 0

    def _start_commit(self):
        """커밋 개시: fork_remaining 을 채워 step 이 turn_bias 로 꺾게 한다."""
        self.commit_triggered = True
        self.fork_remaining = self.p['fork_commit_frames']

    def _begin_start(self, resume_from_red=True):
        """출발 개시(초록 확정 또는 빨강 소멸 트리거 공통).

        최초 전이(green_started False→True) 시 조향/분기 래치를 리셋해, 출발
        직후 스테일 커밋으로 꺾이지 않고 직진/차선중앙에서 시작하게 한다. 출발
        직진 유예·킥스타트도 이때 arm 한다. resume_from_red=True 면 빨간불 정지
        (red_stopped)를 해제한다 — 출발선 빨강이 남아 throttle 을 0 으로 누르지
        않게 한다(빨강 소멸 출발은 항상 해제, 초록 출발은 green_resumes_from_red).
        """
        p = self.p
        if not self.green_started:
            self.turn_intent = None
            self.fork_remaining = 0
            self.commit_triggered = False
            self._sign_max_prox = 0.0
            self._sign_absent_streak = 0
            self._sign_seen_since_latch = 0
            self._left_streak = 0
            self._right_streak = 0
            self.last_steer = p['steer_trim']
            self.last_offset = 0.0
            self.corner_hold = 0.0
            self.start_straight_remaining = p['start_straight_frames']
            self.start_kick_remaining = p['start_kick_frames']
        self.green_started = True
        if resume_from_red:
            self.red_stopped = False
            self._red_streak = 0

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
        best_box = {}   # 클래스별 최고점 검출 객체(표지판 박스 기하 참조용)
        for d in detections:
            cls = d.class_id
            if cls == GREENLIGHT and not self.green_started:
                thr = p['green_start_conf']
            else:
                thr = conf
            if d.score < thr:
                continue
            # 신호등(빨강/초록)은 상단 ROI 로 제한(신호등은 트랙 위쪽).
            if cls in (REDLIGHT, GREENLIGHT) and not self._light_in_roi(d, frame_height, frame_width):
                continue
            if cls == REDLIGHT and not self._redlight_is_real(d, frame_height):
                continue
            if d.score > best.get(cls, 0.0):
                best[cls] = d.score
                best_box[cls] = d

        # 초록/빨강 스트릭 → 래치.
        self._green_streak = self._green_streak + 1 if GREENLIGHT in best else 0
        self._red_streak = self._red_streak + 1 if REDLIGHT in best else 0
        # 초록 확정 출발: 출발 순간(최초 전이)에 조향/분기 래치를 리셋하고
        # green_started 래치. 초록 재확정 시 빨간불 정지 해제는 green_resumes_from_red
        # 를 따른다(도착 영구정지 유지 옵션). 상세는 _begin_start.
        if self._green_streak >= p['start_confirm_frames']:
            self._begin_start(resume_from_red=p['green_resumes_from_red'])
        if self._red_streak >= p['stop_confirm_frames']:
            self.red_stopped = True

        # 빨강 소멸 출발(대안 트리거, pre-start): 대기 중 빨강이 확정 정지(red_stopped)
        # 될 만큼 확실히 잡힌 뒤, 빨강이 red_gone_frames 연속 사라지면 = 초록 점등으로
        # 보고 출발한다(빨강은 강하게 검출되므로 약한 초록보다 신뢰↑). 빨강 재검출 시
        # 카운터 리셋(간헐 깜빡 무시). green_started 후에는 동작 안 함(출발선 전용).
        if (p['start_on_red_gone'] and not self.green_started
                and self.red_stopped):
            if REDLIGHT in best:
                self._red_gone_streak = 0
            else:
                self._red_gone_streak += 1
                if self._red_gone_streak >= p['red_gone_frames']:
                    self._begin_start(resume_from_red=True)

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
            latched = None
            if self._left_streak >= p['confirm_frames']:
                latched = 'left'
            elif self._right_streak >= p['confirm_frames']:
                latched = 'right'
            if latched is not None:
                self.turn_intent = latched
                # 커밋 트리거 상태를 래치 시점 기준으로 초기화(근접/소실/백스톱).
                self._sign_max_prox = 0.0
                self._sign_absent_streak = 0
                self._sign_seen_since_latch = 0
        elif not self.commit_triggered:
            # 아직 커밋(꺾기) 전이면 반대 방향이 sign_revise_frames 연속 확정될 때
            # 래치를 갱신한다. 출발 직후 잘못 래치된 방향을 실제 갈림길 표지판이
            # 교정하게 해 오주행을 막는다. 갱신 시 근접 추적을 리셋해 새 방향의
            # 표지판으로 커밋 판정을 다시 시작한다. 커밋 이후엔 방향을 고정(안전).
            if (self.turn_intent == 'right'
                    and self._left_streak >= p['sign_revise_frames']):
                self.turn_intent = 'left'
                self._reset_sign_tracking()
            elif (self.turn_intent == 'left'
                    and self._right_streak >= p['sign_revise_frames']):
                self.turn_intent = 'right'
                self._reset_sign_tracking()

        # 방향 래치와 실제 커밋(꺾기)의 분리: 방향은 위에서 멀리서도 일찍 래치하되,
        # fork_remaining(=꺾기 시작)은 표지판이 가까워졌을 때만 세팅한다. 멀리서
        # 확정하자마자 꺾어 코스를 벗어나는 것을 막는다(B.3). frame_height 미전달
        # 시엔 종전대로 래치 즉시 커밋(_maybe_trigger_commit 참조, 하위호환).
        if self.turn_intent is not None and not self.commit_triggered:
            self._maybe_trigger_commit(best_box, frame_height, frame_width)

    # ---- 제어 timer rate 로 호출 ----
    def step(self, lane):
        """lane: LaneSignal. 반환 (steering, throttle), 둘 다 [-1,1]/[0,1] 범위."""
        p = self.p
        trim = p['steer_trim']

        # --- 조향: 차선 PD (유효 시) 또는 마지막 조향 유지(로스트 폴백) ---
        # trim 기준 상대량(effort)만 계산해 두고, 커밋 여부에 따라 차선 기여도를
        # 조절한다. lane_effort = (차선추종 목표 - trim).
        if lane.valid:
            # 데드밴드를 먼저 먹여 중앙 근처 미세 offset 노이즈를 0 으로 만든다.
            # 비례항뿐 아니라 미분항도 이 값 기준으로 계산해, 직선(±deadband 안)에서는
            # 조향 변화가 정확히 0 이 되게 한다(미분항이 노이즈에 반응해 떠는 것 방지).
            # deadband 를 넘는 실제 드리프트·커브에는 PD·곡률 피드포워드가 그대로 반응.
            p_offset = _deadband(lane.offset, p['steer_deadband'])
            d_off = p_offset - self.last_offset
            self.last_offset = p_offset
            # PD(현재 오차) + 곡률 피드포워드(다가오는 커브 예측). curvature 는
            # offset 과 같은 부호 규약(먼 밴드가 오른쪽으로 휘면 +)이라 steer_sign
            # 만 곱한다 — drive_direction 미러링 대상 아님(offset PD 와 동일).
            lane_effort = p['steer_sign'] * (
                p['steer_kp'] * p_offset
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

        # 트림 기준 대칭 한계로 클램프(과조향 제거). trim=0 이면 [-1,1] 와 동일.
        target = _clamp(target, self._steer_lo, self._steer_hi)
        # 커밋 중엔 분기를 신속히 완성하도록 슬루 상한을 완화(commit_steer_slew).
        slew_limit = p['commit_steer_slew'] if committing else p['steer_slew']
        steer = _clamp(_slew(self.last_steer, target, slew_limit),
                       self._steer_lo, self._steer_hi)

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
            'commit_triggered': self.commit_triggered,
            'fork_remaining': self.fork_remaining,
            'sign_seen': self._sign_seen_since_latch,
            'corner_hold': round(self.corner_hold, 3),
            'start_kick_remaining': self.start_kick_remaining,
            'red_gone': self._red_gone_streak,
        }
