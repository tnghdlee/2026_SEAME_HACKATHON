"""동적 장애물 정지/재출발 상태머신 (B.4).

프로젝트 배치 위치: src/inference/inference/aruco_stop_policy.py

이 파일은 cv2/ROS 에 의존하지 않는 순수 상태머신이다 (driving_policy.py 와 같은 계층).
입력은 매 프레임 "장애물 마커가 보였는가?"(bool) 뿐이고, 출력은 "지금 달려도 되는가?"(bool).

B.4 규정 반영:
- 장애물 등장 시 정지 / 퇴거 시 출발. 정지 중에는 스탑워치가 멈추므로 시간 손해가 없다.
  → 따라서 정지는 '보수적(민감하게, 오검출 허용)', 재출발은 '보수적(소멸을 확실히 확인)'.
- 이 둘의 '보수성 방향'이 반대라는 것이 9.5 의 핵심. 그래서 임계 프레임 수를 비대칭으로:
    stop_confirm_frames  (정지 진입) : 작게  — 몇 프레임만 보여도 즉시 정지
    clear_confirm_frames (재출발)    : 크게  — 연속으로 확실히 사라져야 출발
- 정지 중 시간 손해가 없으므로, 애매하면 계속 정지해 있는 쪽이 항상 이득 (충돌/이탈 회피 우선).

hysteresis(히스테리시스) 구조라 마커가 검출 경계에서 깜빡여도 상태가 진동하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StopPolicyConfig:
    # 정지 진입: 최근 프레임 중 '검출' 이 이만큼 연속되면 STOP. 작게(민감).
    stop_confirm_frames: int = 2
    # 재출발: '미검출' 이 이만큼 연속되어야 RUN 복귀. 크게(보수적).
    clear_confirm_frames: int = 6


class ArucoStopPolicy:
    """마커 present(bool) 스트림 → 주행 허용(bool) 상태머신.

    상태:
      RUN  : 주행 허용 (throttle_scale=1.0)
      STOP : 정지      (throttle_scale=0.0)

    사용:
      policy = ArucoStopPolicy(StopPolicyConfig())
      allow = policy.update(present)   # 매 프레임 호출
      if not allow: throttle = 0.0     # inference_node 에서 throttle 을 덮어씀
    """

    RUN = "RUN"
    STOP = "STOP"

    def __init__(self, config: StopPolicyConfig | None = None):
        self.cfg = config or StopPolicyConfig()
        self.state = self.RUN
        self._present_streak = 0   # 연속 검출 카운터
        self._absent_streak = 0    # 연속 미검출 카운터

    def reset(self) -> None:
        self.state = self.RUN
        self._present_streak = 0
        self._absent_streak = 0

    def update(self, present: bool) -> bool:
        """한 프레임 갱신. 반환값 = 주행 허용 여부(True=달려도 됨)."""
        if present:
            self._present_streak += 1
            self._absent_streak = 0
        else:
            self._absent_streak += 1
            self._present_streak = 0

        if self.state == self.RUN:
            # 정지 진입은 민감하게
            if self._present_streak >= self.cfg.stop_confirm_frames:
                self.state = self.STOP
        else:  # STOP
            # 재출발은 소멸을 확실히 확인한 뒤에만
            if self._absent_streak >= self.cfg.clear_confirm_frames:
                self.state = self.RUN

        return self.state == self.RUN

    @property
    def throttle_scale(self) -> float:
        """주행 정책 throttle 에 곱할 계수 (RUN=1.0, STOP=0.0)."""
        return 1.0 if self.state == self.RUN else 0.0

    @property
    def is_stopped(self) -> bool:
        return self.state == self.STOP