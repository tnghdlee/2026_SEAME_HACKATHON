"""aruco_stop_policy 순수 상태머신 단위 테스트 (ROS/cv2 비의존).

`python3 test/test_aruco_stop_policy.py` 또는 `pytest` 로 실행. 동적 장애물(B.4)
정지/재출발의 비대칭 히스테리시스(9.5: 정지 진입 민감·재출발 보수)를 검증한다.
검출측(aruco_detect)은 cv2.aruco 의존이라 별도(보드에서 검증) — 이 테스트는 순수
상태머신만 다룬다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference.aruco_stop_policy import (  # noqa: E402
    ArucoStopPolicy, StopPolicyConfig,
)


def test_initial_state_is_run():
    p = ArucoStopPolicy(StopPolicyConfig())
    assert p.state == ArucoStopPolicy.RUN
    assert not p.is_stopped
    assert p.throttle_scale == 1.0
    assert p.update(False) is True, '미검출이면 계속 주행 허용'


def test_stop_enters_on_confirm_frames():
    p = ArucoStopPolicy(StopPolicyConfig(stop_confirm_frames=2,
                                         clear_confirm_frames=6))
    # 1프레임 검출로는 아직 정지 안 함(민감하지만 단발 오검출 흡수).
    assert p.update(True) is True
    assert not p.is_stopped
    # 2프레임 연속 검출 → 정지.
    assert p.update(True) is False
    assert p.is_stopped
    assert p.throttle_scale == 0.0


def test_resume_requires_clear_confirm_frames():
    p = ArucoStopPolicy(StopPolicyConfig(stop_confirm_frames=2,
                                         clear_confirm_frames=6))
    p.update(True)
    p.update(True)          # STOP 진입
    assert p.is_stopped
    # 미검출이 clear_confirm_frames(6) 미만이면 계속 정지(보수적 재출발).
    for _ in range(5):
        assert p.update(False) is False
        assert p.is_stopped
    # 6번째 연속 미검출 → 재출발.
    assert p.update(False) is True
    assert not p.is_stopped


def test_flicker_does_not_resume():
    # 정지 중 마커가 깜빡여도(검출↔미검출) 미검출 연속이 끊기면 재출발하지 않는다.
    p = ArucoStopPolicy(StopPolicyConfig(stop_confirm_frames=2,
                                         clear_confirm_frames=3))
    p.update(True)
    p.update(True)          # STOP
    for _ in range(10):
        p.update(False)     # 미검출 2회(임계 3 미만)
        p.update(False)
        p.update(True)      # 다시 검출 → 미검출 스트릭 리셋
        assert p.is_stopped, '깜빡임 중엔 계속 정지(히스테리시스)'


def test_stop_is_more_sensitive_than_resume():
    # 비대칭성: 정지 임계(2) < 재출발 임계(6). 정지는 빨리, 재출발은 느리게.
    cfg = StopPolicyConfig(stop_confirm_frames=2, clear_confirm_frames=6)
    assert cfg.stop_confirm_frames < cfg.clear_confirm_frames


def test_reset_returns_to_run():
    p = ArucoStopPolicy(StopPolicyConfig(stop_confirm_frames=1))
    p.update(True)          # STOP(임계 1)
    assert p.is_stopped
    p.reset()
    assert p.state == ArucoStopPolicy.RUN
    assert not p.is_stopped


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'PASS {fn.__name__}')
    print(f'=== {len(fns)} tests PASS ===')


if __name__ == '__main__':
    _run_all()
