"""ArUco 마커 검출 (동적 장애물 B.4).

프로젝트 배치 위치: src/inference/inference/aruco_detect.py

설계 원칙 (CLAUDE.md 9번, B.4, OT 17p 트랙 도식):
- 이 파일은 cv2 에만 의존하고 ROS/rclpy 에는 의존하지 않는다 (opencv/lane_detect.py 와 같은 계층).
  -> 합성 이미지로 단위 테스트 가능. ROS 배선은 inference_node.py 가 담당.
- 마커 판별 결과(대회 제공 이미지 실측): DICT_6X6_50, ID=3. 사전/ID 는 param 으로 주입.
  ※ OT 슬라이드에는 사전/ID 명시가 없어, '별도 제공된 마커 이미지 실측값'을 근거로 삼음.
- 3중 게이팅으로 '정지해야 할 마커'만 통과시킨다:
    (1) target_ids    : 규정 마커 ID 만 인정 (기본 (3,))
    (2) ROI           : 화면 하단 중앙(내 주행 경로) 안에 마커 중심이 있을 때만  <- OT 트랙 도식 반영
    (3) min_area_ratio: 화면 점유 면적(근접도)이 임계 이상일 때만 (멀면 무시)
  트랙 옆/반대편/원거리 마커에 오정지하는 것을 막기 위함.

이 모듈은 "이번 프레임에 (경로상·충분히 가까운) 규정 마커가 있는가?" boolean 만 만든다.
정지/재출발 타이밍(시간적 필터링)은 aruco_stop_policy.py 상태머신이 담당한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np


_DICT_MAP = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_6X6_50": aruco.DICT_6X6_50,    # <- 대회 마커 (실측 확정)
    "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_7X7_50": aruco.DICT_7X7_50,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
}


@dataclass
class ArucoConfig:
    """검출 파라미터. 전부 ROS param 으로 노출할 것 (하드코딩 금지 - CLAUDE.md 9-4)."""
    dict_name: str = "DICT_6X6_50"
    # 규정 마커 ID. 제공된 마커가 ID=3 하나뿐이므로 기본을 (3,) 로 특화.
    # 대회가 여러 ID 를 쓰면 확장하고, ID 무관하게 아무 마커나 잡으려면 None.
    target_ids: Optional[Tuple[int, ...]] = (3,)
    # 근접 게이팅: 마커 bbox 면적 / 프레임 면적 이 이 값 이상일 때만 '가까운 장애물'로 인정.
    # 0.0 이면 게이팅 없음. 실측 튜닝 대상.
    min_area_ratio: float = 0.0
    # ROI(관심영역) 게이팅: 마커 '중심'이 이 정규화 사각형 안에 있을 때만 인정.
    # (x0, y0, x1, y1), 각 0~1. 기본 = 화면 하단 60%, 가로 중앙 60% (내 주행 경로).
    # None 이면 전체 화면. OT 트랙 도식상 마커는 주행 경로 위에 등장.
    roi_norm: Optional[Tuple[float, float, float, float]] = (0.2, 0.4, 0.8, 1.0)


def create_detector(config: ArucoConfig) -> "aruco.ArucoDetector":
    """OpenCV 4.7+ 신 API 로 Detector 생성.

    주의: 보드 opencv-python-headless 5.x / 개발 cv2 4.13 모두 신 API 대상.
    구 API(aruco.Dictionary_get / aruco.detectMarkers(img, dict, ...)) 는 쓰지 말 것.
    """
    if config.dict_name not in _DICT_MAP:
        raise ValueError(
            f"unknown dict_name={config.dict_name!r}; choose one of {sorted(_DICT_MAP)}"
        )
    dictionary = aruco.getPredefinedDictionary(_DICT_MAP[config.dict_name])
    params = aruco.DetectorParameters()
    return aruco.ArucoDetector(dictionary, params)


@dataclass
class MarkerHit:
    marker_id: int
    area_ratio: float                 # 프레임 대비 면적 (근접도 프록시)
    center: Tuple[float, float]       # 픽셀 좌표
    center_norm: Tuple[float, float]  # 0~1 정규화 좌표 (ROI 판정용)
    corners: np.ndarray               # (4,2)


def detect_markers(gray: np.ndarray, detector: "aruco.ArucoDetector") -> List[MarkerHit]:
    """그레이스케일 프레임에서 마커를 검출해 MarkerHit 리스트로 반환 (게이팅 전 원시 결과)."""
    if gray is None or gray.size == 0:
        return []
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    frame_area = float(h * w) if h * w else 1.0

    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return []

    hits: List[MarkerHit] = []
    for c, i in zip(corners, ids.flatten()):
        pts = c.reshape(4, 2).astype(np.float32)
        area = float(cv2.contourArea(pts))
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        hits.append(
            MarkerHit(
                marker_id=int(i),
                area_ratio=area / frame_area,
                center=(cx, cy),
                center_norm=(cx / w, cy / h),
                corners=pts,
            )
        )
    return hits


def _in_roi(cn: Tuple[float, float], roi: Optional[Tuple[float, float, float, float]]) -> bool:
    if roi is None:
        return True
    x0, y0, x1, y1 = roi
    x, y = cn
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def obstacle_present(gray: np.ndarray, detector: "aruco.ArucoDetector",
                     config: ArucoConfig) -> Tuple[bool, List[MarkerHit]]:
    """이번 프레임에 '정지해야 할 장애물 마커'가 있는지 판정 (3중 게이팅).

    반환: (present, hits)
      present : target_ids + ROI + min_area_ratio 를 모두 통과한 마커가 하나라도 있으면 True
      hits    : 진단/로깅용 전체 검출 목록 (게이팅 전)

    주의: present 는 '이 한 프레임'의 즉시 판정. 실제 정지/재출발은 단발 오검출에 흔들리지 않도록
      aruco_stop_policy.ArucoStopPolicy 에 넘겨 연속 프레임 조건으로 확정할 것
      (B.4: 정지 보수적 / 재출발은 소멸 확인 후).
    """
    hits = detect_markers(gray, detector)
    for hit in hits:
        if config.target_ids is not None and hit.marker_id not in config.target_ids:
            continue
        if not _in_roi(hit.center_norm, config.roi_norm):
            continue
        if hit.area_ratio < config.min_area_ratio:
            continue
        return True, hits
    return False, hits