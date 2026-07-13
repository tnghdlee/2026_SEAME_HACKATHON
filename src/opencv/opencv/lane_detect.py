"""규칙 기반 차선 오프셋 추정 (CLAUDE.md 8.1 — 차선 추종), Hough 라인 방식.

설계:
    이 모듈은 하단 ROI 에서 차선 마킹을 이진화한 뒤 *확률적 허프 변환*
    (`cv2.HoughLinesP`)으로 직선 세그먼트를 뽑아 좌/우 차선을 피팅한다.
    각 라인을 x = f(y) (거의 수직) 로 보고, 가까운 y(near, ROI 하단)와 먼 y(far,
    ROI 상단)에서의 x 를 외삽해 차선 중앙을 구한다. near 의 중앙이 횡오차(offset),
    near→far 중앙 변화가 곡률(curvature) 이다.

    좌/우 분리(split_lanes):
      - split_lanes=True (양쪽 경계선 트랙): 세그먼트를 화면 중앙(원본 w/2 를 ROI
        로컬로 투영한 분할선) 기준 좌/우로 나눠 각각 평균 피팅하고, 두 라인의
        중점을 차선 중앙으로 삼는다. 한쪽만 보이면 그 라인 ± 차폭(lane_half_norm)
        으로 반대쪽을 추정한다.
      - split_lanes=False (단일 중앙선, 예: 주황 라인): 검출된 모든 라인을 하나로
        평균해 그 라인 자체를 추종한다.

    이진화 경로(method)는 밴드-무게중심 시절과 동일하게 유지한다 — 트랙 프로파일
    (brightness/light vs color/dark) 이 그대로 동작하도록:
      - brightness : 그레이스케일 + adaptiveThreshold(polarity)
      - color      : HSV inRange 색 마스크
    이렇게 만든 이진 마스크에서 Canny 에지를 뽑아 HoughLinesP 를 돌린다.

    ⚠️ Hough 방식의 알려진 한계(밴드-무게중심 대비):
      1) 점선·짧은 마킹은 minLineLength/maxLineGap 튜닝에 민감하다. 마킹이 끊기면
         세그먼트가 안 잡혀 valid=False 가 될 수 있다(밴드 방식은 질량만 있으면 됨).
      2) 급커브에서 직선 가정이 깨진다 — near/far 2점 외삽이라 곡선을 직선으로
         근사한다. 곡률 추정은 near vs far 기울기 차의 근사치일 뿐이다.
      3) 광택 바닥의 반사/주름이 에지로 잡혀 가짜 라인을 만들 수 있다. 각도
         게이트(hough_min_angle_deg)로 near-수평 세그먼트(정지선/노이즈)를 버린다.
    실트랙 튜닝은 Hough 파라미터(threshold/min_line_length/max_line_gap/
    min_angle_deg/canny_low/high)로 하며, 전부 opencv_node 의 ROS param 이다.

이 모듈은 순수 함수(ROS 비의존)라 오프라인에서 단위 테스트·튜닝이 가능하다.

반환하는 LaneResult 필드:
    offset      [-1, 1]로 정규화된 횡오차. <0 = 차선 중앙이 원본 이미지
                중앙(w/2, ROI 크롭과 무관)보다 왼쪽, >0 = 오른쪽. 부호→조향
                매핑은 하류(steer_sign)에서 적용.
    valid       Hough 라인이 잡혀 offset 을 신뢰할 수 있으면 True.
    curvature   [-1, 1] 곡률 추정치. (먼_offset - 가까운_offset)/2. |curvature|가
                클수록 다가오는 커브가 급함. 부호는 커브 방향.
    pixels      이진 마스크의 차선 픽셀 총수(ROI 전체) — 로깅/튜닝용.
    valid_bands 차선 중앙 산출에 기여한 라인 side 수(0/1/2). 하류가 신뢰도 판단에
                사용(1이면 한쪽만 보여 반대쪽을 추정한 것 → 근거가 약함).
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LaneResult:
    offset: float
    valid: bool
    curvature: float
    pixels: int = 0        # 이진 마스크의 차선 픽셀 총수(ROI 전체) — 로깅/튜닝용.
    valid_bands: int = 0   # 차선 중앙에 기여한 라인 side 수 — 신뢰도 판단용.


# 극성(polarity): 차선 표시가 노면보다 밝은가 어두운가?
POLARITY_LIGHT = 'light'  # 어두운 바닥 위 밝은 테이프/도색
POLARITY_DARK = 'dark'    # 밝은 바닥 위 어두운 라인

# 검출 방식: 'brightness'=그레이스케일 명암(폴라리티), 'color'=HSV 색 마스크.
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


def _binarize_lane(gray, polarity, block_size=25):
    """adaptive threshold → 차선 픽셀=255 인 이진 마스크.

    (전역 Otsu 가 아니라) adaptive 를 쓰는 이유: ROI 전반의 불균일한 조명·그림자
    때문에 한쪽 차선이 통째로 지워지지 않게 하기 위함.

    block_size 는 지역 평균을 구하는 이웃 창의 픽셀 크기이며 반드시 홀수여야
    한다(짝수/1 이하면 가장 가까운 유효 홀수로 보정). 이 값은 해상도에 비례해
    스케일해야 한다.
    """
    block_size = int(block_size)
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_type = (cv2.THRESH_BINARY if polarity == POLARITY_LIGHT
                   else cv2.THRESH_BINARY_INV)
    mask = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        thresh_type,
        blockSize=block_size,
        C=-10 if polarity == POLARITY_LIGHT else 10,
    )
    return mask


def _denoise(mask, ksize=3):
    """형태학적 열림(MORPH_OPEN: 침식→팽창)으로 산발 노이즈를 제거한다.

    ksize<=1 이면 그대로 반환(비활성). Hough 이전에 점 노이즈를 줄여 가짜
    에지·세그먼트를 억제한다.
    """
    if ksize is None or ksize <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _norm_offset(cx_local, roi_left, img_half):
    """ROI 로컬 열 인덱스 → 원본 이미지 중앙 기준 정규화 오프셋 [-1,1].

    roi_left 를 더해 원본 이미지 좌표로 되돌린 뒤 (x - img_half)/img_half 로
    정규화하므로, 비대칭 ROI(roi_left>0) 여도 offset=0 이 항상 원본 이미지
    중앙(카메라 중심선)을 가리킨다. img_half = 원본 이미지 폭 w / 2.
    """
    return (roi_left + cx_local - img_half) / img_half


def _line_x_at(seg, y):
    """세그먼트 (x1,y1,x2,y2) 를 x=f(y) 직선으로 보고 주어진 y 에서의 x 를 외삽.

    수평(y1==y2) 세그먼트는 수직 외삽이 불가능하므로 None 반환(호출 전에 각도
    게이트로 이미 걸러지지만 0 나눗셈 방어).
    """
    x1, y1, x2, y2 = seg
    if y2 == y1:
        return None
    m = (x2 - x1) / float(y2 - y1)
    return x1 + m * (y - y1)


def _avg_x(points, idx):
    """(x_near, x_far) 튜플 리스트에서 idx(0=near,1=far) 평균."""
    n = len(points)
    if n == 0:
        return None
    return sum(p[idx] for p in points) / float(n)


def compute_lane_offset(
    image_bgr,
    roi_top=50,
    roi_left=0,
    roi_right=None,
    num_bands=3,          # (호환용, Hough 방식에선 미사용)
    valid_min_px=40,
    polarity=POLARITY_LIGHT,
    method=METHOD_BRIGHTNESS,
    hsv_lower=(5, 80, 80),
    hsv_upper=(22, 255, 255),
    morph_ksize=3,
    split_lanes=False,
    lane_half_norm=0.5,
    side_min_px=None,     # (호환용, Hough 방식에선 미사용)
    block_size=25,
    hough_threshold=30,
    hough_min_line_length=20,
    hough_max_line_gap=15,
    hough_min_angle_deg=25.0,
    canny_low=50,
    canny_high=150,
):
    """BGR 프레임에서 Hough 라인으로 정규화된 횡방향 차선 오프셋을 추정.

    파라미터는 opencv_node 의 ROS param 과 대응 — 튜닝을 코드가 아닌 설정에서
    하도록 함(CLAUDE.md 9.4). 반환 필드는 모듈 독스트링 참조.

    이진화(method/polarity/color/split_lanes)는 밴드-무게중심 시절과 동일해
    트랙 프로파일이 그대로 동작한다. 그 이진 마스크에서 Canny 에지를 뽑아
    HoughLinesP 로 세그먼트를 검출하고 좌/우 차선을 피팅한다.

    num_bands / side_min_px 는 밴드-무게중심 방식의 잔여 인자로 Hough 에선
    사용하지 않지만, 기존 호출부(opencv_node / 튜닝 하니스) 호환을 위해 시그니처를
    유지한다.
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
        mask = _binarize_lane(gray, polarity, block_size)

    mask = _denoise(mask, morph_ksize)

    # ROI 전체에서 검출된 라인 픽셀 총수 — valid_min_px 튜닝/로깅용.
    total_px = int((mask > 0).sum())

    roi_h, roi_w = mask.shape[:2]
    img_half = w / 2.0
    # 원본 이미지 중앙을 ROI 로컬 좌표로 투영한 좌/우 분할선.
    split_local = int(round(img_half - roi_left))
    split_local = max(0, min(split_local, roi_w))

    # 마스크 픽셀이 너무 적으면 라인 신뢰 불가(조기 종료).
    if total_px < valid_min_px:
        return LaneResult(0.0, False, 0.0, pixels=total_px, valid_bands=0)

    # 이진 마스크 → 에지 → 확률적 허프 변환.
    edges = cv2.Canny(mask, int(canny_low), int(canny_high))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(hough_threshold),
        minLineLength=int(hough_min_line_length),
        maxLineGap=int(hough_max_line_gap),
    )

    if lines is None:
        return LaneResult(0.0, False, 0.0, pixels=total_px, valid_bands=0)

    # near = ROI 하단(가장 가까움), far = ROI 상단(가장 멈). y 는 ROI 로컬(위=0).
    y_near = roi_h - 1
    y_far = 0

    left = []       # [(x_near, x_far), ...]  분할선 왼쪽 라인
    right = []      # 분할선 오른쪽 라인
    all_lines = []  # split_lanes=False 용 전체

    for ln in lines:
        # HoughLinesP 반환 shape 는 버전에 따라 (N,1,4) 또는 (N,4) — flatten 해서 통일.
        vals = np.asarray(ln, dtype=np.float64).reshape(-1)
        x1, y1, x2, y2 = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
        dx = x2 - x1
        dy = y2 - y1
        # 각도 게이트: near-수평 세그먼트(정지선/노이즈)는 버림. 차선은 거의 수직.
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        if angle < hough_min_angle_deg:
            continue
        seg = (x1, y1, x2, y2)
        xn = _line_x_at(seg, y_near)
        xf = _line_x_at(seg, y_far)
        if xn is None or xf is None:
            continue
        all_lines.append((xn, xf))
        # near 위치로 좌/우 분류(분할선 기준).
        if xn < split_local:
            left.append((xn, xf))
        else:
            right.append((xn, xf))

    lane_half_px = lane_half_norm * img_half  # 정규화 반차폭 → 픽셀

    center_near = None
    center_far = None
    valid_bands = 0

    if split_lanes:
        ln_near, ln_far = _avg_x(left, 0), _avg_x(left, 1)
        rn_near, rn_far = _avg_x(right, 0), _avg_x(right, 1)
        if ln_near is not None and rn_near is not None:
            center_near = (ln_near + rn_near) / 2.0
            center_far = (ln_far + rn_far) / 2.0
            valid_bands = 2
        elif ln_near is not None:
            center_near = ln_near + lane_half_px
            center_far = ln_far + lane_half_px
            valid_bands = 1
        elif rn_near is not None:
            center_near = rn_near - lane_half_px
            center_far = rn_far - lane_half_px
            valid_bands = 1
    else:
        an_near, an_far = _avg_x(all_lines, 0), _avg_x(all_lines, 1)
        if an_near is not None:
            center_near = an_near
            center_far = an_far
            valid_bands = 1

    if center_near is None:
        return LaneResult(0.0, False, 0.0, pixels=total_px, valid_bands=0)

    off_near = _norm_offset(center_near, roi_left, img_half)
    off_far = _norm_offset(center_far, roi_left, img_half)

    # 현재 위치(near)에 더 큰 가중치, 먼 지점(far)은 약한 피드포워드 반영.
    offset = float(np.clip((2.0 * off_near + off_far) / 3.0, -1.0, 1.0))
    # 곡률: near→far 중앙 변화. 원시 범위 [-2,2] → /2 정규화.
    curvature = float(np.clip((off_far - off_near) / 2.0, -1.0, 1.0))

    return LaneResult(
        offset, True, curvature,
        pixels=total_px, valid_bands=valid_bands,
    )
