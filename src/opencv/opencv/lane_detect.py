"""규칙 기반 차선 오프셋 추정 (CLAUDE.md 8.1 TODO — 차선 추종).

설계 (inference_node.compute_control 의 설계 주석 참조):
    Hough 2-라인 피팅은 점선·그림자·급커브에서 쉽게 깨집니다. 완주 우선
    해커톤에서는 대신 *밴드 무게중심(band-centroid)* 방식을 씁니다: 하단 ROI를
    몇 개의 수평 밴드로 나누고, 각 밴드에서 차선 픽셀을 이진화한 뒤 차선 질량의
    가로 무게중심을 구합니다. 그 무게중심이 이미지 중앙에서 얼마나 벗어났는지가
    횡오차이며, 가까운 밴드와 먼 밴드를 비교하면 값싼 곡률(피드포워드) 추정치를
    얻습니다.

    ⚠️ 단일 무게중심의 함정(split_lanes): 밴드 전체 열에 대해 단일 무게중심을
    구하면, 좌우 두 라인이 동시에 잡힐 때 무게중심이 두 라인 사이의 *빈 노면*을
    가리킵니다. D-Racer 트랙처럼 양쪽 경계선이 있을 수 있는 구조에서는
    `split_lanes=True` 로 좌/우 질량을 나눠 각 라인을 따로 구하고 그 중점을
    차선 중앙으로 삼습니다. 한쪽만 보이면 그 라인 + 차폭(lane_half_norm)으로
    반대쪽을 추정합니다. 단일 라인(예: 주황 중앙선)만 추종할 때는
    `split_lanes=False`.

    ⚠️ split_lanes 의 한계 (실측 데이터 없이 코드로 확정하지 말 것):
      1) 고정 분할선 — 분할선은 '원본 이미지 중앙(w/2)'을 ROI 로컬 좌표로
         투영한 고정 열이다. 급커브에서 좌/우 경계선이 모두 화면 같은 반쪽으로
         몰리면 한쪽 side 로 병합돼 오검출한다. 적응형(질량 골짜기 기반) 분할선은
         실차 데이터 확보 후 도입할 것. 현재는 valid_bands(유효 밴드 수)와
         '한쪽-only 폴백'을 반환해 하류(inference_node)가 신뢰도를 낮출 수
         있게만 한다.
      2) per-side 픽셀 게이트 — 좌/우 각각 side_min_px(기본 valid_min_px/2)를
         통과한 side 만 신뢰한다. 한쪽 노이즈 몇 px 가 phantom 무게중심으로
         채택돼 차선 중앙을 끌어당기는 것을 차단한다.
      3) lane_half_norm — 한쪽 라인만 보일 때 반대쪽을 추정하는 정규화 반차폭.
         기본 0.5 = '차선 반폭 = 이미지 반폭의 0.5'(≈ 차선 폭이 이미지 폭의 절반)
         가정이다. 흰 경계선 트랙에서 한쪽 소실 구간의 조향 정확도를 좌우하므로
         실측 튜닝 대상.

이 모듈은 순수 함수(ROS 비의존)라 오프라인에서 단위 테스트·튜닝이 가능합니다.

반환하는 LaneResult 필드:
    offset      [-1, 1]로 정규화된 횡오차. <0 = 차선 중앙이 원본 이미지
                중앙(w/2, ROI 크롭과 무관)보다 왼쪽(차는 좌조향해야 함),
                >0 = 오른쪽. 부호→조향 매핑은 하류(steer_sign)에서 적용.
    valid       `offset`을 신뢰할 만큼 차선 픽셀이 충분히 검출됐으면 True.
    curvature   [-1, 1]로 정규화된 곡률 추정치. (먼_밴드_offset - 가까운_밴드_offset)를
                /2 하여 정규화(원시 차이 범위 [-2, 2] → [-1, 1]). |curvature| 가 클수록
                다가오는 커브가 급함. 0 = 직진(밴드 간 오프셋 차 없음). 부호는 커브 방향.
    pixels      검출된 차선 픽셀 총수(ROI 전체) — 로깅/튜닝용.
    valid_bands 오프셋 산출에 실제 기여한 유효 밴드 수. 하류가 신뢰도 판단에 사용
                (1개뿐이면 곡률=0 이고 오프셋 근거가 약함).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LaneResult:
    offset: float
    valid: bool
    curvature: float
    pixels: int = 0        # 검출된 차선 픽셀 총수(ROI 전체) — 로깅/튜닝용.
    valid_bands: int = 0   # 오프셋에 기여한 유효 밴드 수 — 신뢰도 판단용.


# 극성(polarity): 차선 표시가 노면보다 밝은가 어두운가?
POLARITY_LIGHT = 'light'  # 어두운 바닥 위 밝은 테이프/도색
POLARITY_DARK = 'dark'    # 밝은 바닥 위 어두운 라인

# 검출 방식: 'brightness'=그레이스케일 명암(폴라리티), 'color'=HSV 색 마스크.
# 색 라인(예: 흰 바닥 위 주황 라인)은 'color' 가 훨씬 강건하다 — 그레이스케일은
# 광택 바닥의 반사·주름을 라인으로 오검출한다(실측 확인).
#
# ⚠️ 이 함수의 method 기본값은 'brightness' 지만, opencv_node 는 ROS param
#    `lane_method` 로 이를 'color' 로 덮어쓴다(CLAUDE.md 10번). 순수 함수를
#    단독 테스트할 때는 트랙 라인 색에 맞는 method 를 명시적으로 넘길 것.
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


def _denoise(mask, ksize=3):
    """형태학적 열림(MORPH_OPEN: 침식→팽창)으로 산발 노이즈를 제거한다.

    inRange/adaptiveThreshold 직후의 점 노이즈가 valid_min_px 를 우연히 넘겨
    거짓 밴드를 만드는 것을 막는다. ksize<=1 이면 그대로 반환(비활성).

    ⚠️ open 은 라인을 '메우지' 않는다(닫힘 아님). 오히려 얇은 먼-밴드 흰 선은
    침식 단계에서 지워져 먼 밴드가 무효가 되고 곡률이 자주 0 이 될 수 있다.
    실트랙 흰 선 두께 실측 후 MORPH_CLOSE 병행/커널 크기 확정은 튜닝 대상이며,
    지금은 기본 동작(open, ksize=3)을 유지한다.
    """
    if ksize is None or ksize <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened


def _norm_offset(cx_local, roi_left, img_half):
    """ROI 로컬 열 인덱스 → 원본 이미지 중앙 기준 정규화 오프셋 [-1,1].

    roi_left 를 더해 원본 이미지 좌표로 되돌린 뒤 (x - img_half)/img_half 로
    정규화하므로, 비대칭 ROI(roi_left>0) 여도 offset=0 이 항상 원본 이미지
    중앙(카메라 중심선)을 가리킨다. img_half = 원본 이미지 폭 w / 2.
    """
    return (roi_left + cx_local - img_half) / img_half


def _band_centroid_single(col_mass, roi_left, img_half):
    """밴드 전체 열 질량의 단일 무게중심 → 정규화 오프셋. 실패 시 (None, 0.0).

    정규화 기준은 '원본 이미지 중앙'(img_half=w/2) — _norm_offset 참조.
    """
    total = float(col_mass.sum())
    if total <= 0.0:
        return None, 0.0
    cols = np.arange(col_mass.shape[0], dtype=np.float32)
    centroid_local = float((cols * col_mass).sum() / total)
    return _norm_offset(centroid_local, roi_left, img_half), total


def _band_centroid_split(col_mass, roi_left, img_half, lane_half_norm, side_min_px):
    """좌/우 질량을 나눠 각 라인 무게중심을 구하고 그 중점을 차선 중앙으로.

    - 분할선: 원본 이미지 중앙(img_half)을 ROI 로컬 좌표로 투영한 고정 열.
    - per-side 게이트: 좌/우 각각 side_min_px 픽셀 이상 통과한 side 만 신뢰
      (한쪽 노이즈 몇 px 가 phantom 무게중심으로 채택되는 것을 차단).
    - 좌우 모두 통과: 중점 = (left + right)/2.
    - 한쪽만 통과: 그 라인에서 차폭(lane_half_norm)만큼 안쪽으로 이동해 추정.
      (예: 왼쪽만 보이면 center = left_off + lane_half_norm)
    - 둘 다 미통과: (None, 0.0) → 해당 밴드 무효.
    정규화 기준은 _band_centroid_single 과 동일(원본 이미지 중앙).
    반환: (정규화 오프셋 or None, 채택된 side 들의 총 질량)
    """
    roi_w = col_mass.shape[0]
    # 원본 이미지 중앙을 ROI 로컬 좌표로 투영해 분할선으로 사용.
    split_local = int(round(img_half - roi_left))
    split_local = max(0, min(split_local, roi_w))
    left_mass = col_mass[:split_local]
    right_mass = col_mass[split_local:]

    def side_offset(mass, base_idx):
        t = float(mass.sum())
        if t < side_min_px:      # per-side 게이트: 노이즈성 소량 픽셀 배제
            return None, 0.0
        c = np.arange(mass.shape[0], dtype=np.float32)
        cx_local = float((c * mass).sum() / t) + base_idx
        return _norm_offset(cx_local, roi_left, img_half), t

    left_off, left_t = side_offset(left_mass, 0)
    right_off, right_t = side_offset(right_mass, split_local)

    if left_off is not None and right_off is not None:
        return (left_off + right_off) / 2.0, left_t + right_t
    if left_off is not None:
        return left_off + lane_half_norm, left_t
    if right_off is not None:
        return right_off - lane_half_norm, right_t
    return None, 0.0


def compute_lane_offset(
    image_bgr,
    roi_top=50,
    roi_left=0,
    roi_right=None,
    num_bands=3,
    valid_min_px=40,
    polarity=POLARITY_LIGHT,
    method=METHOD_BRIGHTNESS,
    hsv_lower=(5, 80, 80),
    hsv_upper=(22, 255, 255),
    morph_ksize=3,
    split_lanes=False,
    lane_half_norm=0.5,
    side_min_px=None,
):
    """BGR 프레임에서 정규화된 횡방향 차선 오프셋을 추정.

    파라미터는 opencv_node 의 ROS param 과 대응 — 튜닝을 코드가 아닌 설정에서
    하도록 함 (CLAUDE.md 9.4). 반환 필드는 모듈 독스트링 참조.

    method='color' 이면 HSV 색 마스크(hsv_lower~hsv_upper)로 라인을 검출한다.
    흰/회색 바닥 위 유색 라인에는 이쪽이 강건하다. 'brightness' 는 그레이스케일
    명암(polarity) 방식.

    split_lanes=True 이면 좌/우 라인을 분리해 그 중점을 차선 중앙으로 삼는다
    (양쪽 경계선 트랙). False 이면 단일 무게중심(단일 중앙선 추종).
    lane_half_norm 은 한쪽 라인만 보일 때 반대쪽을 추정하기 위한 정규화 반차폭.
    side_min_px 는 split 모드의 per-side 픽셀 게이트(None → valid_min_px/2).

    오프셋 정규화 기준은 항상 '원본 이미지 중앙(w/2)' 이다 — 비대칭 ROI
    (roi_left>0/roi_right<w) 에서도 offset=0 이 카메라 중심선을 뜻한다.
    """
    if image_bgr is None or image_bgr.size == 0:
        return LaneResult(0.0, False, 0.0)

    h, w = image_bgr.shape[:2]
    roi_top = int(max(0, min(roi_top, h - 1)))
    roi_left = int(max(0, min(roi_left, w - 1)))
    roi_right = w if roi_right is None else int(max(roi_left + 1, min(roi_right, w)))

    roi = image_bgr[roi_top:h, roi_left:roi_right]
    if roi.size == 0:
        return LaneResult(0.0, False, 0.0)

    if method == METHOD_COLOR:
        mask = _color_mask(roi, hsv_lower, hsv_upper)
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = _binarize_lane(gray, polarity)

    mask = _denoise(mask, morph_ksize)

    # ROI 전체에서 검출된 라인 픽셀 총수 — valid_min_px 튜닝/로깅용.
    total_px = int((mask > 0).sum())

    # 오프셋 정규화 기준 = 원본 이미지 중앙(w/2). ROI 중앙이 아니라 원본 중앙을
    # 써야 비대칭 ROI 에서도 offset=0 이 카메라 중심선을 가리킨다.
    img_half = w / 2.0
    # split 모드 per-side 게이트: 지정 없으면 밴드 유효 임계의 절반.
    side_gate = (valid_min_px / 2.0) if side_min_px is None else float(side_min_px)

    roi_h, roi_w = mask.shape[:2]
    num_bands = max(1, int(num_bands))
    band_h = max(1, roi_h // num_bands)

    band_offsets = []  # (밴드 인덱스, 정규화 오프셋); 인덱스 0 = 가장 가까움(하단)
    for i in range(num_bands):
        # 밴드 0 이 ROI 의 최하단(가장 가까운) 슬라이스.
        y1 = max(0, roi_h - (i + 1) * band_h)
        y2 = roi_h - i * band_h
        band = mask[y1:y2, :]

        col_mass = band.sum(axis=0).astype(np.float32) / 255.0

        # 밴드 유효성은 '해당 밴드의 총 질량'으로 판정(ROI 전체가 아니라).
        if float(col_mass.sum()) < valid_min_px:
            continue

        if split_lanes:
            off, _ = _band_centroid_split(
                col_mass, roi_left, img_half, lane_half_norm, side_gate)
        else:
            off, _ = _band_centroid_single(col_mass, roi_left, img_half)

        if off is None:
            continue
        band_offsets.append((i, float(off)))

    if not band_offsets:
        return LaneResult(0.0, False, 0.0, pixels=total_px, valid_bands=0)

    # 가까운 밴드에 더 큰 가중치(현재 위치 반영), 먼 밴드는 약한 피드포워드.
    # 가중치 = num_bands - 밴드 인덱스.
    weight_sum = 0.0
    offset_acc = 0.0
    for band_index, off in band_offsets:
        weight = float(num_bands - band_index)
        offset_acc += weight * off
        weight_sum += weight
    offset = offset_acc / weight_sum if weight_sum > 0 else 0.0

    # 곡률: '첫(가장 가까운) 유효 밴드' 대 '마지막(가장 먼) 유효 밴드' 를 명시적으로
    # 집는다. band_offsets 는 인덱스 오름차순(가까움→멈)이므로 [0]=near, [-1]=far.
    # 유효 밴드가 1개뿐이면 곡률 0(비교 대상 없음).
    near_off = band_offsets[0][1]
    far_off = band_offsets[-1][1]
    raw_curvature = far_off - near_off  # 범위 [-2, 2]
    curvature = float(np.clip(raw_curvature / 2.0, -1.0, 1.0))

    offset = float(np.clip(offset, -1.0, 1.0))
    return LaneResult(
        offset, True, curvature,
        pixels=total_px, valid_bands=len(band_offsets),
    )