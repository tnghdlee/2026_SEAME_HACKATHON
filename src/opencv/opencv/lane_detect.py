"""규칙 기반 차선 오프셋 추정 (CLAUDE.md 8.1 TODO — 차선 추종).

설계 (inference_node.compute_control 의 설계 주석 참조):
    Hough 2-라인 피팅은 점선·그림자·급커브에서 쉽게 깨집니다. 완주 우선
    해커톤에서는 대신 *밴드 무게중심(band-centroid)* 방식을 씁니다: 하단 ROI를
    몇 개의 수평 밴드로 나누고, 각 밴드에서 차선 픽셀을 이진화한 뒤 차선 질량의
    가로 무게중심을 구합니다. 그 무게중심이 이미지 중앙에서 얼마나 벗어났는지가
    횡오차이며, 가까운 밴드와 먼 밴드를 비교하면 값싼 곡률(피드포워드) 추정치를
    얻습니다.

이 모듈은 순수 함수(ROS 비의존)라 오프라인에서 단위 테스트·튜닝이 가능합니다.

반환하는 LaneResult 필드:
    offset    [-1, 1]로 정규화된 횡오차. <0 = 차선 중앙이 이미지 중앙보다
              왼쪽(차는 좌조향해야 함), >0 = 오른쪽. 부호→조향 매핑은
              하류(steer_sign)에서 적용.
    valid     `offset`을 신뢰할 만큼 차선 픽셀이 충분히 검출됐으면 True.
    curvature [-1, 1]로 정규화된 곡률 추정치. (먼_밴드_offset - 가까운_밴드_offset)를
              /2 하여 정규화(원시 차이 범위 [-2, 2] → [-1, 1]). |curvature| 가 클수록
              다가오는 커브가 급함. 0 = 직진(밴드 간 오프셋 차 없음). 부호는 커브 방향.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LaneResult:
    offset: float
    valid: bool
    curvature: float
    pixels: int = 0  # 검출된 차선 픽셀 총수(가장 가까운 유효 밴드 기준) — 로깅/튜닝용.


# 극성(polarity): 차선 표시가 노면보다 밝은가 어두운가?
POLARITY_LIGHT = 'light'  # 어두운 바닥 위 밝은 테이프/도색
POLARITY_DARK = 'dark'    # 밝은 바닥 위 어두운 라인

# 검출 방식: 'brightness'=그레이스케일 명암(폴라리티), 'color'=HSV 색 마스크.
# 색 라인(예: 흰 바닥 위 주황 라인)은 'color' 가 훨씬 강건하다 — 그레이스케일은
# 광택 바닥의 반사·주름을 라인으로 오검출한다(실측 확인).
METHOD_BRIGHTNESS = 'brightness'
METHOD_COLOR = 'color'


def _color_mask(roi_bgr, hsv_lower, hsv_upper):
    """HSV inRange 로 특정 색(예: 주황) 라인만 남긴 이진 마스크.

    hsv_lower/upper 는 (H, S, V) 튜플. OpenCV H 범위는 0~180 이다
    (주황≈5~22, 노랑≈22~38). 저채도(흰/회색 바닥)는 S 하한으로 걸러진다.
    """
    blur = cv2.GaussianBlur(roi_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _binarize_lane(gray, polarity):
    """adaptive threshold → 차선 픽셀=255 인 이진 마스크.

    (전역 Otsu 가 아니라) adaptive 를 쓰는 이유: ROI 전반의 불균일한 조명·그림자
    때문에 한쪽 차선이 통째로 지워지지 않게 하기 위함.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_type = (cv2.THRESH_BINARY if polarity == POLARITY_LIGHT
                   else cv2.THRESH_BINARY_INV)
    mask = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        thresh_type,
        blockSize=25,
        C=-10 if polarity == POLARITY_LIGHT else 10,
    )
    return mask


def compute_lane_offset(
    image_bgr,
    roi_top=50,
    roi_left=0,
    num_bands=3,
    valid_min_px=40,
    polarity=POLARITY_LIGHT,
    method=METHOD_BRIGHTNESS,
    hsv_lower=(5, 80, 80),
    hsv_upper=(22, 255, 255),
):
    """BGR 프레임에서 정규화된 횡방향 차선 오프셋을 추정.

    파라미터는 opencv_node 의 ROS param 과 대응 — 튜닝을 코드가 아닌 설정에서
    하도록 함 (CLAUDE.md 9.4). 반환 필드는 모듈 독스트링 참조.

    method='color' 이면 HSV 색 마스크(hsv_lower~hsv_upper)로 라인을 검출한다.
    흰/회색 바닥 위 유색 라인에는 이쪽이 강건하다. 'brightness' 는 기존
    그레이스케일 명암(polarity) 방식.
    """
    if image_bgr is None or image_bgr.size == 0:
        return LaneResult(0.0, False, 0.0)

    h, w = image_bgr.shape[:2]
    roi_top = int(max(0, min(roi_top, h - 1)))
    roi_left = int(max(0, min(roi_left, w - 1)))

    roi = image_bgr[roi_top:h, roi_left:w]
    if roi.size == 0:
        return LaneResult(0.0, False, 0.0)

    if method == METHOD_COLOR:
        mask = _color_mask(roi, hsv_lower, hsv_upper)
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = _binarize_lane(gray, polarity)

    # ROI 전체에서 검출된 라인 픽셀 총수 — valid_min_px 튜닝/로깅용.
    total_px = int((mask > 0).sum())

    roi_h, roi_w = mask.shape[:2]
    half_w = roi_w / 2.0
    num_bands = max(1, int(num_bands))
    band_h = max(1, roi_h // num_bands)

    # 열 인덱스 벡터 — 밴드마다 질량 가중 무게중심 계산에 재사용.
    cols = np.arange(roi_w, dtype=np.float32)

    band_offsets = []  # (밴드 인덱스, 정규화 오프셋); 인덱스 0 = 가장 가까움(하단)
    for i in range(num_bands):
        # 밴드 0 이 ROI 의 최하단(가장 가까운) 슬라이스.
        y1 = roi_h - (i + 1) * band_h
        y2 = roi_h - i * band_h
        y1 = max(0, y1)
        band = mask[y1:y2, :]

        col_mass = band.sum(axis=0).astype(np.float32) / 255.0
        total = float(col_mass.sum())
        if total < valid_min_px:
            continue

        centroid_x = float((cols * col_mass).sum() / total)
        band_offsets.append((i, (centroid_x - half_w) / half_w))

    if not band_offsets:
        return LaneResult(0.0, False, 0.0, pixels=total_px)

    # 가까운 밴드에 더 큰 가중치(현재 위치 반영), 먼 밴드는 약한 피드포워드.
    # 가중치 = num_bands - 밴드 인덱스.
    weight_sum = 0.0
    offset_acc = 0.0
    near_off = None
    far_off = None
    for band_index, off in band_offsets:
        weight = float(num_bands - band_index)
        offset_acc += weight * off
        weight_sum += weight
        if band_index == band_offsets[0][0]:
            near_off = off
        far_off = off  # 마지막으로 채택된 밴드 = 가장 먼 유효 밴드

    offset = offset_acc / weight_sum if weight_sum > 0 else 0.0

    # 곡률 = 먼_밴드_offset - 가까운_밴드_offset. 밴드 오프셋은 각각 [-1, 1] 이므로
    # 원시 차이 범위는 [-2, 2]. /2 로 [-1, 1] 정규화해 코너 임계를 스케일-안정적으로
    # (offset 과 같은 [-1,1] 단위로) 비교할 수 있게 한다. 순수 횡이동(직선 차선이
    # 중앙에서 벗어난 경우)은 near≈far → curvature≈0 이라 위치와 무관한 곡률 지표다.
    raw_curvature = (far_off - near_off) if (near_off is not None and far_off is not None) else 0.0
    curvature = float(np.clip(raw_curvature / 2.0, -1.0, 1.0))

    offset = float(np.clip(offset, -1.0, 1.0))
    return LaneResult(offset, True, curvature, pixels=total_px)
