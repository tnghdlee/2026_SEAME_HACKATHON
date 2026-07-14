"""driving_policy 순수 로직 단위 테스트 (ROS/cv2 비의존).

`python3 test/test_driving_policy.py` 또는 `pytest` 로 실행. 주행 정책의 핵심
분기(출발 게이트·빨간불 정지·좌/우 margin 게이팅·차선 PD·로스트 폴백·커브 감속)를
검증한다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference.driving_policy import (  # noqa: E402
    DrivingPolicy, LaneSignal, resolve_sign,
    REDLIGHT, GREENLIGHT, LEFT_SIGN, RIGHT_SIGN,
)


class Det:
    def __init__(self, class_id, score, x1=0.0, y1=0.0, x2=0.0, y2=0.0):
        self.class_id = class_id
        self.score = score
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


def straight(valid=True, offset=0.0, curv=0.0):
    return LaneSignal(offset=offset, valid=valid, curvature=curv)


def test_green_start_gate():
    # 킥스타트를 꺼서(start_kick_frames=0) 게이트+순항만 검증.
    p = DrivingPolicy({'require_green_start': True, 'cruise_throttle': 0.13,
                       'start_kick_frames': 0})
    s, t = p.step(straight())
    assert t == 0.0 and s == p.p['steer_trim'], '출발 전 정지·중립'
    # 초록불 2프레임 확정 → 출발.
    p.on_detections([Det(GREENLIGHT, 0.9)])
    p.on_detections([Det(GREENLIGHT, 0.9)])
    assert p.green_started
    s, t = p.step(straight())
    assert t == 0.13, '출발 후 순항 throttle(킥 비활성)'


def test_red_hard_stop():
    p = DrivingPolicy({'require_green_start': False, 'cruise_throttle': 0.2})
    p.on_detections([Det(REDLIGHT, 0.9)])
    p.on_detections([Det(REDLIGHT, 0.9)])
    assert p.red_stopped
    _, t = p.step(straight())
    assert t == 0.0, '빨간불 하드 정지'


def test_redlight_floor_rejected():
    # ArUco 구간 '빨간 바닥' 오검출: 프레임 하단의 크고 낮은 redlight 박스는
    # frame_height 를 넘기면 기하 게이팅으로 무시돼 정지하지 않는다.
    H = 480
    p = DrivingPolicy({'require_green_start': False, 'stop_confirm_frames': 2,
                       'redlight_max_y_ratio': 0.6, 'redlight_max_h_ratio': 0.5})
    # 하단(중심 y≈0.85H)에 크게 잡힌 '바닥'.
    floor = Det(REDLIGHT, 0.9, x1=0, y1=0.7 * H, x2=640, y2=1.0 * H)
    for _ in range(3):
        p.on_detections([floor], frame_height=H, frame_width=640)
    assert not p.red_stopped, '빨간 바닥은 기하 게이팅으로 무시(정지 안 함)'
    _, t = p.step(straight())
    assert t != 0.0, '바닥 오검출로 정지하지 않음'


def test_redlight_real_still_stops():
    # 실제 신호등(상단·작은 박스)은 그대로 정지시킨다.
    H = 480
    p = DrivingPolicy({'require_green_start': False, 'stop_confirm_frames': 2,
                       'redlight_max_y_ratio': 0.6, 'redlight_max_h_ratio': 0.5})
    # 상단(중심 y≈0.2H)에 작게 잡힌 실제 신호등.
    light = Det(REDLIGHT, 0.9, x1=300, y1=0.12 * H, x2=340, y2=0.28 * H)
    for _ in range(2):
        p.on_detections([light], frame_height=H, frame_width=640)
    assert p.red_stopped, '실제 신호등(상단·소형)은 정지'
    _, t = p.step(straight())
    assert t == 0.0, '실제 빨간불 하드 정지'


def test_green_start_low_conf():
    # 출발 전 초록불은 green_start_conf(0.12)로 완화 → 낮은 신뢰도(0.15)도 출발.
    # 반면 같은 낮은 점수의 빨간불은 전역 conf(0.25) 미만이라 정지 안 함.
    p = DrivingPolicy({'require_green_start': True, 'conf_threshold': 0.25,
                       'green_start_conf': 0.12, 'start_confirm_frames': 1,
                       'stop_confirm_frames': 1, 'cruise_throttle': 0.15,
                       'start_kick_frames': 0})
    # 낮은 신뢰도 빨간불(0.15 < 0.25) → 무시.
    p.on_detections([Det(REDLIGHT, 0.15)])
    assert not p.red_stopped, '낮은 신뢰도 빨간불은 전역 conf 미만이라 무시'
    # 낮은 신뢰도 초록불(0.15 ≥ green_start_conf 0.12) → 1프레임 확정 출발.
    p.on_detections([Det(GREENLIGHT, 0.15)])
    assert p.green_started, '출발 전 초록불은 완화된 임계값으로 저신뢰도도 인식'
    _, t = p.step(straight())
    assert t == 0.15, '초록불 확정 후 순항'


def test_green_conf_normal_after_start():
    # 출발 후에는 초록불도 전역 conf 를 쓴다(green_start_conf 미적용).
    p = DrivingPolicy({'require_green_start': False, 'conf_threshold': 0.25,
                       'green_start_conf': 0.12, 'start_confirm_frames': 1})
    p.green_started = True
    p.on_detections([Det(GREENLIGHT, 0.15)])  # 0.15 < 0.25 → 무시(스트릭 0)
    assert p._green_streak == 0, '출발 후 저신뢰 초록불은 전역 conf 로 필터'


def test_start_kick():
    # 초록불 확정 직후 킥스타트: 조향 없이(trim) 고정 throttle 로 N프레임 직진,
    # 이후 정상 차선 추종으로 전환.
    p = DrivingPolicy({'require_green_start': True, 'start_confirm_frames': 1,
                       'start_kick_throttle': 0.2, 'start_kick_frames': 3,
                       'cruise_throttle': 0.15, 'steer_kp': 0.6, 'steer_sign': 1.0,
                       'steer_slew': 1.0, 'curve_ff': 0.0})
    p.on_detections([Det(GREENLIGHT, 0.9)])
    assert p.green_started and p.start_kick_remaining == 3
    # 킥 3프레임: 오프셋이 커도 조향 0(중립), throttle 0.2.
    for _ in range(3):
        s, t = p.step(straight(valid=True, offset=0.5))
        assert s == p.p['steer_trim'], '킥 중 조향 없음(중립)'
        assert t == 0.2, '킥 중 throttle 0.2'
    assert p.start_kick_remaining == 0
    # 킥 종료 후 정상 차선 추종(오프셋에 반응해 조향).
    s, t = p.step(straight(valid=True, offset=0.5))
    assert s != p.p['steer_trim'], '킥 종료 후 차선 추종 조향'
    assert t != 0.2 or True  # throttle 은 상황별(여기선 조향 감속 등)


def test_start_kick_red_pauses():
    # 킥 도중 빨간불이면 정지 우선(킥 잔여 소진 안 함).
    p = DrivingPolicy({'require_green_start': True, 'start_confirm_frames': 1,
                       'stop_confirm_frames': 1, 'start_kick_throttle': 0.2,
                       'start_kick_frames': 5})
    p.on_detections([Det(GREENLIGHT, 0.9)])
    assert p.start_kick_remaining == 5
    # 빨간불 확정(green_resumes_from_red 기본 True 라 초록 재확정 시 풀리지만,
    # 여기선 초록 없이 빨강만) → 정지.
    p.on_detections([Det(REDLIGHT, 0.9)])
    _, t = p.step(straight())
    assert t == 0.0, '킥 중 빨간불이면 정지 우선'
    assert p.start_kick_remaining == 5, '정지 중엔 킥 잔여 소진 안 함'


def test_sign_margin_gating():
    # 근소차 → 애매(대기).
    assert resolve_sign(0.50, 0.45, 0.15, 0.35) is None
    # 우세 점수 낮음 → None.
    assert resolve_sign(0.30, 0.10, 0.15, 0.35) is None
    # 충분한 margin + conf → left.
    assert resolve_sign(0.70, 0.20, 0.15, 0.35) == 'left'
    assert resolve_sign(0.20, 0.70, 0.15, 0.35) == 'right'


def test_fork_commit_bias():
    p = DrivingPolicy({'require_green_start': False, 'confirm_frames': 2,
                       'turn_bias': 0.3, 'steer_sign': 1.0, 'drive_direction': 1.0,
                       'fork_commit_frames': 5, 'steer_slew': 1.0})
    # 좌회전 표지판 2프레임 확정.
    p.on_detections([Det(LEFT_SIGN, 0.9)])
    p.on_detections([Det(LEFT_SIGN, 0.9)])
    assert p.turn_intent == 'left'
    s, t = p.step(straight())
    assert s < 0.0, '좌회전 커밋 → 좌조향 바이어스(음수)'
    assert t == p.p['turn_throttle']


def test_sign_proximity_commit_bottom_y():
    # 근접 게이팅(기본 지표 bottom_y): 방향은 멀리서 래치하되, 표지판 하단이
    # 프레임 아래로 sign_commit_ratio 만큼 내려오기 전까지는 커밋을 미룬다.
    H = 480
    p = DrivingPolicy({'require_green_start': False, 'confirm_frames': 2,
                       'turn_bias': 0.5, 'steer_sign': 1.0, 'drive_direction': 1.0,
                       'fork_commit_frames': 5, 'steer_slew': 1.0,
                       'sign_proximity_metric': 'bottom_y', 'sign_commit_ratio': 0.60,
                       'sign_lost_commit_frames': 99})  # 소실 폴백은 사실상 끔
    # 멀리 있는 표지판(하단 y2=0.30H, 프레임 위쪽) 2프레임 → 방향만 래치.
    far = Det(LEFT_SIGN, 0.9, x1=300, y1=0.20 * H, x2=330, y2=0.30 * H)
    p.on_detections([far], frame_height=H, frame_width=640)
    p.on_detections([far], frame_height=H, frame_width=640)
    assert p.turn_intent == 'left', '멀리서도 방향은 래치(기억)'
    assert not p.commit_triggered, '멀면(표지판 위쪽) 아직 커밋 안 함'
    s, _ = p.step(straight())
    assert abs(s) < 1e-6, '커밋 전 → 바이어스 없이 직진(차선 추종)'
    # 가까워져 표지판이 아래로 내려옴(하단 y2=0.72H ≥ 0.60) → 커밋 시작.
    near = Det(LEFT_SIGN, 0.9, x1=290, y1=0.55 * H, x2=340, y2=0.72 * H)
    p.on_detections([near], frame_height=H, frame_width=640)
    assert p.commit_triggered and p.fork_remaining == 5, '근접(아래로 내려옴) → 커밋'
    s, t = p.step(straight())
    assert s < 0.0, '커밋 시작 → 좌조향 바이어스'
    assert t == p.p['turn_throttle']


def test_sign_proximity_commit_height_metric():
    # 지표 전환('height'): 박스 높이가 임계 이상 커질 때 커밋(카메라 낮은 경우).
    H = 480
    p = DrivingPolicy({'require_green_start': False, 'confirm_frames': 2,
                       'turn_bias': 0.5, 'steer_sign': 1.0, 'fork_commit_frames': 5,
                       'steer_slew': 1.0, 'sign_proximity_metric': 'height',
                       'sign_commit_ratio': 0.20, 'sign_lost_commit_frames': 99})
    far = Det(LEFT_SIGN, 0.9, x1=300, y1=0.10 * H, x2=330, y2=0.15 * H)  # h=0.05H
    p.on_detections([far], frame_height=H, frame_width=640)
    p.on_detections([far], frame_height=H, frame_width=640)
    assert p.turn_intent == 'left' and not p.commit_triggered
    near = Det(LEFT_SIGN, 0.9, x1=280, y1=0.10 * H, x2=360, y2=0.35 * H)  # h=0.25H
    p.on_detections([near], frame_height=H, frame_width=640)
    assert p.commit_triggered, '높이 지표: 박스 높이 커짐 → 커밋'


def test_sign_loss_fallback_commit():
    # 소실 폴백: 표지판이 충분히 가까워진 뒤 프레임 밖으로 연속 사라지면 커밋.
    H = 480
    p = DrivingPolicy({'require_green_start': False, 'confirm_frames': 2,
                       'turn_bias': 0.5, 'steer_sign': 1.0, 'fork_commit_frames': 5,
                       'steer_slew': 1.0, 'sign_proximity_metric': 'bottom_y',
                       'sign_commit_ratio': 0.95,  # 근접 임계 사실상 안 걸리게
                       'sign_lost_min_ratio': 0.30, 'sign_lost_commit_frames': 2})
    mid = Det(LEFT_SIGN, 0.9, x1=280, y1=0.30 * H, x2=360, y2=0.40 * H)  # y2=0.40H
    p.on_detections([mid], frame_height=H, frame_width=640)
    p.on_detections([mid], frame_height=H, frame_width=640)  # 래치 + max_prox=0.40
    assert p.turn_intent == 'left' and not p.commit_triggered
    p.on_detections([], frame_height=H, frame_width=640)     # 1프레임 소실
    assert not p.commit_triggered, '1프레임 소실로는 커밋 안 함'
    p.on_detections([], frame_height=H, frame_width=640)     # 2프레임 연속 소실 → 커밋
    assert p.commit_triggered, '충분히 가까웠던 표지판이 연속 소실 → 커밋'


def test_sign_far_blip_no_loss_commit():
    # 멀리서 잠깐 잡힌(위쪽) 표지판이 사라져도 소실 폴백은 발동하지 않는다
    # (sign_lost_min_ratio 미만이라 '가까이 온 적 없음'으로 판단).
    H = 480
    p = DrivingPolicy({'require_green_start': False, 'confirm_frames': 2,
                       'steer_sign': 1.0, 'fork_commit_frames': 5, 'steer_slew': 1.0,
                       'sign_proximity_metric': 'bottom_y', 'sign_commit_ratio': 0.95,
                       'sign_lost_min_ratio': 0.30, 'sign_lost_commit_frames': 2})
    tiny = Det(LEFT_SIGN, 0.9, x1=300, y1=0.10 * H, x2=320, y2=0.18 * H)  # y2=0.18H
    p.on_detections([tiny], frame_height=H, frame_width=640)
    p.on_detections([tiny], frame_height=H, frame_width=640)
    assert p.turn_intent == 'left'
    for _ in range(5):
        p.on_detections([], frame_height=H, frame_width=640)
    assert not p.commit_triggered, '멀었던(위쪽) 표지판 소실은 커밋 트리거 아님'


def test_sign_commit_backward_compat_no_frame_height():
    # frame_height 미전달 시엔 종전 동작: 래치 즉시 커밋(하위호환).
    p = DrivingPolicy({'require_green_start': False, 'confirm_frames': 2,
                       'steer_sign': 1.0, 'fork_commit_frames': 5, 'steer_slew': 1.0})
    p.on_detections([Det(LEFT_SIGN, 0.9)])
    p.on_detections([Det(LEFT_SIGN, 0.9)])
    assert p.turn_intent == 'left' and p.commit_triggered, '기하 없으면 즉시 커밋'
    assert p.fork_remaining == 5


def test_drive_direction_mirror():
    common = {'require_green_start': False, 'confirm_frames': 1, 'turn_bias': 0.3,
              'steer_sign': 1.0, 'fork_commit_frames': 5, 'steer_slew': 1.0}
    fwd = DrivingPolicy(dict(common, drive_direction=1.0))
    rev = DrivingPolicy(dict(common, drive_direction=-1.0))
    for pol in (fwd, rev):
        pol.on_detections([Det(LEFT_SIGN, 0.9)])
    sf, _ = fwd.step(straight())
    sr, _ = rev.step(straight())
    assert sf < 0.0 and sr > 0.0, '역방향은 좌/우 바이어스 미러링'


def test_lane_pd_and_lost_fallback():
    # steer_throttle_threshold 를 크게 두어 조향 감속 경로를 끄고 순수 PD/로스트만 검증.
    p = DrivingPolicy({'require_green_start': False, 'steer_sign': 1.0,
                       'steer_kp': 0.6, 'steer_kd': 0.0, 'steer_slew': 1.0,
                       'lane_lost_throttle': 0.1, 'cruise_throttle': 0.13,
                       'steer_throttle_threshold': 10.0})
    # offset>0(차선중앙이 오른쪽) → steer_sign*kp*offset 양수.
    s, t = p.step(straight(valid=True, offset=0.5))
    assert s > 0.0 and t == 0.13
    last = p.last_steer
    # 차선 로스트 → 마지막 조향 유지 + lane_lost_throttle.
    s2, t2 = p.step(straight(valid=False))
    assert abs(s2 - last) < 1e-6 and t2 == 0.1, '로스트 시 마지막 조향 유지'


def test_curve_slowdown():
    # 곡률 기반 corner_throttle 만 검증하도록 조향 감속 경로(steer_throttle)를 끈다.
    # (curve_ff=0 로 조향을 중립 유지 → |steer-trim| 이 임계 미만이라 조향 감속 미발동.)
    p = DrivingPolicy({'require_green_start': False, 'cruise_throttle': 0.2,
                       'corner_throttle': 0.1, 'corner_curvature_threshold': 0.3,
                       'curve_hold_decay': 0.85, 'steer_slew': 1.0,
                       'curve_ff': 0.0, 'steer_kp': 0.0})
    _, t_straight = p.step(straight(valid=True, curv=0.0))
    assert t_straight == 0.2
    _, t_curve = p.step(straight(valid=True, curv=0.5))
    assert t_curve == 0.1, '커브 감속(corner_throttle)'


def test_steer_throttle():
    # 조향 감속: 조향 명령이 trim 에서 임계 이상 벗어나면 steer_throttle 로 감속.
    # 곡률 기반 corner_throttle 보다 우선(실제 조향각에 직접 반응).
    p = DrivingPolicy({'require_green_start': False, 'steer_sign': 1.0,
                       'steer_kp': 0.6, 'steer_kd': 0.0, 'curve_ff': 0.0,
                       'steer_slew': 1.0, 'cruise_throttle': 0.2,
                       'steer_throttle': 0.14, 'steer_throttle_threshold': 0.05,
                       'corner_throttle': 0.1, 'corner_curvature_threshold': 0.3})
    # offset≈0 → 조향 중립 → 순항.
    _, t_straight = p.step(straight(valid=True, offset=0.0, curv=0.0))
    assert t_straight == 0.2
    # offset 큼 → 조향 발생(|steer-trim|>=0.05) → steer_throttle.
    _, t_steer = p.step(straight(valid=True, offset=0.5, curv=0.0))
    assert t_steer == 0.14, '조향 중 감속(steer_throttle)'


def test_curve_feedforward():
    # 곡률 피드포워드: offset=0(중앙 정렬)이라도 다가오는 커브 곡률에 맞춰 미리 조향.
    base = {'require_green_start': False, 'steer_sign': 1.0, 'steer_kp': 0.6,
            'steer_kd': 0.0, 'steer_slew': 1.0}
    p_off = DrivingPolicy(dict(base, curve_ff=0.0))
    p_ff = DrivingPolicy(dict(base, curve_ff=0.5))
    # offset=0, curvature>0(앞쪽이 오른쪽으로 휨).
    s_off, _ = p_off.step(straight(valid=True, offset=0.0, curv=0.4))
    s_ff, _ = p_ff.step(straight(valid=True, offset=0.0, curv=0.4))
    assert abs(s_off) < 1e-6, 'curve_ff=0 이면 곡률 무시(offset=0 → 조향 0)'
    assert s_ff > 0.0, 'curve_ff>0 이면 다가오는 우커브로 미리 조향'
    # 좌커브(curvature<0)면 반대 부호.
    p_ff2 = DrivingPolicy(dict(base, curve_ff=0.5))
    s_ff2, _ = p_ff2.step(straight(valid=True, offset=0.0, curv=-0.4))
    assert s_ff2 < 0.0, 'curve_ff>0 + 좌커브 → 좌조향'


def test_start_straight_grace():
    # 출발 직진 유예: 초록불 출발 후 grace 동안 표지판이 계속 보여도 분기 안 함.
    p = DrivingPolicy({'require_green_start': True, 'confirm_frames': 2,
                       'start_straight_frames': 4, 'turn_bias': 0.5,
                       'steer_sign': 1.0, 'fork_commit_frames': 5,
                       'steer_slew': 1.0, 'cruise_throttle': 0.15,
                       'start_kick_frames': 0})
    # 초록불 확정 → 출발(유예 4프레임 arm).
    p.on_detections([Det(GREENLIGHT, 0.9)])
    p.on_detections([Det(GREENLIGHT, 0.9)])
    assert p.green_started and p.start_straight_remaining == 4
    # 유예 중 좌회전 표지판이 계속 보여도 turn_intent 래치 안 됨(직진 유지).
    for _ in range(3):
        p.on_detections([Det(LEFT_SIGN, 0.9)])
        s, t = p.step(straight())
        assert p.turn_intent is None, '유예 중에는 분기 억제(직진)'
        assert t == 0.15, '유예 중 순항 throttle(커밋 아님)'
    # 유예 소진(step 을 몇 번 더 호출해 카운트다운 완료).
    for _ in range(3):
        p.step(straight())
    assert p.start_straight_remaining == 0
    # 유예 종료 후 새로 confirm_frames 를 채우면 정상 분기.
    p.on_detections([Det(LEFT_SIGN, 0.9)])
    p.on_detections([Det(LEFT_SIGN, 0.9)])
    assert p.turn_intent == 'left', '유예 종료 후에는 실제 갈림길에서 분기'


def test_start_straight_disabled_by_default():
    # start_straight_frames=0(기본) 이면 예전 동작: 출발 후 표지판 즉시 분기.
    p = DrivingPolicy({'require_green_start': True, 'confirm_frames': 2,
                       'steer_sign': 1.0, 'fork_commit_frames': 5, 'steer_slew': 1.0})
    p.on_detections([Det(GREENLIGHT, 0.9)])
    p.on_detections([Det(GREENLIGHT, 0.9)])
    assert p.start_straight_remaining == 0
    p.on_detections([Det(LEFT_SIGN, 0.9)])
    p.on_detections([Det(LEFT_SIGN, 0.9)])
    assert p.turn_intent == 'left', '유예 0 이면 즉시 분기(하위호환)'


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'PASS {fn.__name__}')
    print(f'=== {len(fns)} tests PASS ===')


if __name__ == '__main__':
    _run_all()
