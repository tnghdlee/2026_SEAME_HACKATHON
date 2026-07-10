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
    def __init__(self, class_id, score):
        self.class_id = class_id
        self.score = score


def straight(valid=True, offset=0.0, curv=0.0):
    return LaneSignal(offset=offset, valid=valid, curvature=curv)


def test_green_start_gate():
    p = DrivingPolicy({'require_green_start': True, 'cruise_throttle': 0.13})
    s, t = p.step(straight())
    assert t == 0.0 and s == p.p['steer_trim'], '출발 전 정지·중립'
    # 초록불 2프레임 확정 → 출발.
    p.on_detections([Det(GREENLIGHT, 0.9)])
    p.on_detections([Det(GREENLIGHT, 0.9)])
    assert p.green_started
    s, t = p.step(straight())
    assert t == 0.13, '출발 후 순항 throttle'


def test_red_hard_stop():
    p = DrivingPolicy({'require_green_start': False, 'cruise_throttle': 0.2})
    p.on_detections([Det(REDLIGHT, 0.9)])
    p.on_detections([Det(REDLIGHT, 0.9)])
    assert p.red_stopped
    _, t = p.step(straight())
    assert t == 0.0, '빨간불 하드 정지'


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
    p = DrivingPolicy({'require_green_start': False, 'steer_sign': 1.0,
                       'steer_kp': 0.6, 'steer_kd': 0.0, 'steer_slew': 1.0,
                       'lane_lost_throttle': 0.1, 'cruise_throttle': 0.13})
    # offset>0(차선중앙이 오른쪽) → steer_sign*kp*offset 양수.
    s, t = p.step(straight(valid=True, offset=0.5))
    assert s > 0.0 and t == 0.13
    last = p.last_steer
    # 차선 로스트 → 마지막 조향 유지 + lane_lost_throttle.
    s2, t2 = p.step(straight(valid=False))
    assert abs(s2 - last) < 1e-6 and t2 == 0.1, '로스트 시 마지막 조향 유지'


def test_curve_slowdown():
    p = DrivingPolicy({'require_green_start': False, 'cruise_throttle': 0.2,
                       'corner_throttle': 0.1, 'corner_curvature_threshold': 0.3,
                       'curve_hold_decay': 0.85, 'steer_slew': 1.0})
    _, t_straight = p.step(straight(valid=True, curv=0.0))
    assert t_straight == 0.2
    _, t_curve = p.step(straight(valid=True, curv=0.5))
    assert t_curve == 0.1, '커브 감속(corner_throttle)'


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


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'PASS {fn.__name__}')
    print(f'=== {len(fns)} tests PASS ===')


if __name__ == '__main__':
    _run_all()
