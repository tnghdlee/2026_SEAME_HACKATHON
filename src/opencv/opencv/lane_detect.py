"""규칙 기반 차선 오프셋 추정 (CLAUDE.md 8.1 — 차선 추종), BEV + 슬라이딩 윈도우 방식.

설계:
    직선만 잡던 Hough 방식을 대체한다. 곡선·직선을 모두 인식하기 위해
    Bird's Eye View(조감도) 변환 후 히스토그램으로 좌/우 차선 출발점을 찾고,
    슬라이딩 윈도우로 차선 픽셀을 위로 추적해 좌/우 각각 2차 함수(x = a·y² + b·y + c)
    로 적합한다. 적합 곡선에서 차선 중앙을 구해 near(하단)를 횡오차(offset),
    near→far(상단) 중앙 변화를 곡률(curvature)로 삼는다.

    파이프라인(프레임당):
      1) 이진화        : brightness(adaptive threshold, polarity) 또는 color(HSV inRange)
      2) BEV 원근변환  : 사다리꼴 ROI 4점 → 직사각형(조감도). 4점은 원본 폭/높이 대비
                         0~1 비율(bev_src_*)로 지정 → 해상도 무관, 하드코딩 회피(9.4).
      3) 히스토그램    : BEV 하단부 열 합 → 좌/우 반쪽에서 각각 argmax = 출발 x
      4) 슬라이딩 윈도우: 출발점에서 위로 창을 쌓으며 차선 픽셀 수집(minpix 넘으면 재중심화)
      5) 2차 함수 적합 : 좌/우 각각 x = f(y). 충분한 픽셀이 있을 때만.
      6) offset/curvature: 차선 중앙(near/far)과 카메라 중심의 편차(정규화)

    한쪽 차선 소실(직선 구간 흰선 하나만 보이는 등):
      검출된 라인에서 lane_width_ratio(BEV 폭 대비 차폭)의 절반만큼 옆으로
      이동해 반대쪽을 추정한다(valid_bands=1, 신뢰도 낮음 표시).

    이진화 경로(method):
      - white      : HSV 저채도·고명도 inRange 흰색 마스크          [대회 white_track 기본]
      - brightness : 그레이스케일 + adaptiveThreshold(polarity)  [명암 기반 대안]
      - color      : HSV inRange 색 마스크                        [연습 orange_track]

    white 경로 배경: 밝기(brightness) 기반은 조명이 어두우면 임계가 흔들리고,
    트랙 양옆 파란 매트가 그레이스케일에서 중간 밝기라 마스크에 새어 든다. 흰 선의
    결정적 특징인 "채도(S) 낮음 + 명도(V) 높음"으로 게이팅하면 유채색(파란 매트/
    신발)이 채도로 걸러진다.

    Hough 방식 대비 이점:
      - 급커브를 직선 2점 외삽이 아니라 2차 곡선으로 적합 → 곡률 표현이 실제 곡선을 따름.
      - 점선·끊긴 마킹도 히스토그램/윈도우가 픽셀 질량만으로 추적 → 세그먼트가 끊겨도 강건.

이 모듈은 순수 함수(ROS 비의존)라 오프라인에서 단위 테스트·튜닝이 가능하다.

⚠️ 실차 필수: BEV 4점 캘리브레이션. 직선 구간에서 조감도상 좌우 차선이 세로로
   나란한 평행선이 되도록 bev_src_* 를 맞춰야 offset=0 이 카메라 중심선을 뜻한다.

반환 LaneResult 필드:
    offset      [-1,1] 정규화 횡오차. <0 = 차선 중앙이 카메라 중심선보다 왼쪽,
                >0 = 오른쪽. 부호→조향 매핑은 하류(steer_sign)에서 적용.
    valid       좌/우 중 한쪽 이상 신뢰 검출 시 True.
    curvature   [-1,1] 곡률 추정. (먼_offset - 가까운_offset)/2. 부호=커브 방향.
    pixels      BEV 이진 마스크의 차선 픽셀 총수 — 로깅/튜닝용.
    mask_pixels BEV 변환 전 원본 이진 마스크 픽셀 수(흰색 마스크 등) — 로깅/튜닝용.
    valid_bands 차선 중앙 산출에 기여한 라인 side 수(0/1/2). 1이면 한쪽만 보여
                반대쪽을 추정한 것(근거 약함).
    left_detected / right_detected  좌/우 2차 적합 성공 여부(진단).
    lane_width_px  이번 프레임에서 쓰인 반차폭(near, BEV px). 양쪽 검출 시 실측
                (right-left)/2, 한쪽 소실 시 폴백값. 노드가 EMA 로 기억해 다음
                프레임 폴백(prior_half_px)에 되먹여, 한쪽 소실에도 중앙을 유지한다.
    overlay     draw=True 일 때 BEV 시점 디버그 영상(윈도우/적합곡선, BGR). 아니면 None.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# 극성(polarity): 차선 표시가 노면보다 밝은가 어두운가?
POLARITY_LIGHT = 'light'  # 어두운 바닥 위 밝은 테이프/도색 (대회 흰 경계선)
POLARITY_DARK = 'dark'    # 밝은 바닥 위 어두운 라인

# 검출 방식: 'brightness'=그레이스케일 명암(폴라리티), 'color'=HSV 색 마스크,
# 'white'=HSV 저채도·고명도 흰색 마스크(대회 white_track 기본 — 8.2/10번).
METHOD_BRIGHTNESS = 'brightness'
METHOD_COLOR = 'color'
METHOD_WHITE = 'white'


@dataclass
class LaneResult:
    offset: float
    valid: bool
    curvature: float
    pixels: int = 0             # BEV 이진 마스크의 차선 픽셀 총수 — 로깅/튜닝용.
    mask_pixels: int = 0        # BEV 변환 전 원본 이진 마스크 픽셀 수(흰색 마스크 등) — 로깅/튜닝용.
    valid_bands: int = 0        # 차선 중앙에 기여한 라인 side 수 — 신뢰도 판단용.
    left_detected: bool = False
    right_detected: bool = False
    lane_width_px: float = 0.0  # 이번 프레임 반차폭(near, BEV px) — 폴백 메모리 피드백용.
    overlay: Optional[np.ndarray] = None  # draw=True 일 때만 채워지는 BEV 디버그 영상.


# --------------------------------------------------------------------------- #
# 1) 이진화 (기존 트랙 프로파일 유지)
# --------------------------------------------------------------------------- #
def _color_mask(bgr, hsv_lower, hsv_upper):
    """HSV inRange 로 특정 색(예: 주황) 라인만 남긴 이진 마스크.

    hsv_lower/upper 는 (H,S,V). OpenCV H 범위 0~180 (주황≈5~22).
    저채도(흰/회색 바닥)는 S 하한으로 걸러진다.
    """
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _white_mask(bgr, s_max=50, v_min=150, blur_ksize=5):
    """HSV 저채도·고명도 영역만 남긴 흰색 마스크.

    흰 선은 채도(S) 낮고 명도(V) 높음 → inRange(hsv, (0,0,v_min), (180,s_max,255)).
    파란 매트/신발 등 유채색은 S 가 높아 걸러진다(대회 white_track). s_max 를 낮출수록
    유채색 배제가 강해지고, v_min 을 높일수록 밝은 것만 통과한다.

    기본값(s_max=50, v_min=150)은 실트랙 bag(track_full_20260714_082403) HSV 실측으로
    정한 값이다: 흰 선 S≤12·V≥215, 파란 매트 S≥92, 어두운 노면 V≤118 로 분리되어,
    s_max 는 [12,92] 중앙(50), v_min 은 [118,215] 하단여유(150)에 둬 조명 변화 마진을 둔다.
    """
    k = int(blur_ksize)
    if k >= 3:
        if k % 2 == 0:
            k += 1
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, int(v_min)], dtype=np.uint8)
    upper = np.array([180, int(s_max), 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _binarize_lane(gray, polarity, block_size=25, blur_ksize=5):
    """adaptive threshold → 차선 픽셀=255 인 이진 마스크.

    (전역 Otsu 가 아니라) adaptive 를 쓰는 이유: ROI 전반의 불균일한 조명·그림자
    때문에 한쪽 차선이 통째로 지워지지 않게 하기 위함. block_size 는 홀수여야 하며
    해상도에 비례해 스케일할 것.
    """
    block_size = int(block_size)
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1
    k = int(blur_ksize)
    if k >= 3:
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    thresh_type = (cv2.THRESH_BINARY if polarity == POLARITY_LIGHT
                   else cv2.THRESH_BINARY_INV)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, thresh_type,
        blockSize=block_size, C=-10 if polarity == POLARITY_LIGHT else 10)


def _denoise(mask, ksize=3):
    """형태학적 열림(MORPH_OPEN)으로 산발 노이즈 제거. ksize<=1 이면 비활성."""
    if ksize is None or ksize <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ksize), int(ksize)))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# --------------------------------------------------------------------------- #
# 2) BEV 원근변환
# --------------------------------------------------------------------------- #
def build_perspective(img_w, img_h, src_tl, src_tr, src_br, src_bl, warp_w, warp_h):
    """원본→BEV 변환행렬 M 반환. src_* 는 (폭비율, 높이비율) 튜플."""
    src = np.float32([
        [src_tl[0] * img_w, src_tl[1] * img_h],
        [src_tr[0] * img_w, src_tr[1] * img_h],
        [src_br[0] * img_w, src_br[1] * img_h],
        [src_bl[0] * img_w, src_bl[1] * img_h],
    ])
    dst = np.float32([
        [0, 0], [warp_w - 1, 0], [warp_w - 1, warp_h - 1], [0, warp_h - 1],
    ])
    return cv2.getPerspectiveTransform(src, dst)


# --------------------------------------------------------------------------- #
# 3) 히스토그램으로 좌/우 출발점 찾기
# --------------------------------------------------------------------------- #
def find_lane_bases(binary_bev, hist_ratio=0.5):
    """BEV 하단 hist_ratio 비율의 열 합에서 좌/우 반쪽 argmax = 출발 x. 없으면 -1."""
    h, w = binary_bev.shape[:2]
    y0 = int(h * (1.0 - hist_ratio))
    histogram = np.sum(binary_bev[y0:, :] > 0, axis=0)
    mid = w // 2
    left = histogram[:mid]
    right = histogram[mid:]
    leftx_base = int(np.argmax(left)) if left.any() else -1
    rightx_base = int(np.argmax(right) + mid) if right.any() else -1
    return leftx_base, rightx_base


# --------------------------------------------------------------------------- #
# 4) 슬라이딩 윈도우 탐색
# --------------------------------------------------------------------------- #
def sliding_window_search(binary_bev, x_base, nonzero_x, nonzero_y,
                          n_windows=10, margin=30, minpix=25,
                          overlay=None, color=(0, 255, 0)):
    """한쪽 차선 픽셀의 (선형) 인덱스를 슬라이딩 윈도우로 수집. x_base<0 이면 빈 배열."""
    h, w = binary_bev.shape[:2]
    if x_base < 0:
        return np.empty(0, dtype=np.int64)
    win_h = max(1, h // n_windows)
    x_current = x_base
    lane_inds = []
    for win in range(n_windows):
        y_low = h - (win + 1) * win_h
        y_high = h - win * win_h
        x_low = x_current - margin
        x_high = x_current + margin
        if overlay is not None:
            cv2.rectangle(overlay, (x_low, y_low), (x_high, y_high), color, 1)
        good = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                (nonzero_x >= x_low) & (nonzero_x < x_high)).nonzero()[0]
        lane_inds.append(good)
        if len(good) > minpix:
            x_current = int(np.mean(nonzero_x[good]))
    return np.concatenate(lane_inds) if lane_inds else np.empty(0, dtype=np.int64)


def _poly_x(fit, y):
    return float(fit[0] * y * y + fit[1] * y + fit[2])


# --------------------------------------------------------------------------- #
# 5) 엔트리 함수
# --------------------------------------------------------------------------- #
def compute_lane_offset(
    image_bgr,
    *,
    # --- 이진화 ---
    method=METHOD_BRIGHTNESS,
    polarity=POLARITY_LIGHT,
    hsv_lower=(5, 80, 80),
    hsv_upper=(22, 255, 255),
    white_s_max=50,
    white_v_min=150,
    white_combine=False,
    block_size=25,
    blur_ksize=5,
    morph_ksize=3,
    # --- BEV 4점(원본 폭/높이 대비 0~1 비율) + 결과 크기 ---
    # 실트랙 bag(track_full_20260714_082403) 직선·중앙 구간 캘리브레이션 값(opencv_node 와 일치).
    bev_src_tl=(0.234, 0.62),
    bev_src_tr=(0.766, 0.62),
    bev_src_br=(0.982, 1.00),
    bev_src_bl=(0.018, 1.00),
    warp_w=200,
    warp_h=240,
    # --- 슬라이딩 윈도우 & 유효성 ---
    n_windows=10,
    margin=30,
    minpix=25,
    hist_ratio=0.5,
    min_lane_px=200,
    lane_width_ratio=0.55,
    min_sep_ratio=0.30,
    prior_half_px=None,
    valid_min_px=40,
    draw=False,
):
    """BGR 프레임에서 BEV+슬라이딩 윈도우로 정규화 횡오차·곡률을 추정.

    파라미터는 opencv_node 의 ROS param 과 대응 — 튜닝을 코드가 아닌 설정에서(9.4).
    반환 필드는 모듈 독스트링 참조.

    양쪽 차선 추종 강건화(한쪽만 보고 달리는 증상 방지):
      - min_sep_ratio: 좌/우 둘 다 적합됐을 때 near 간격이 BEV 폭의 이 비율보다
        좁으면 두 윈도우가 같은 물리 라인에 겹쳐 잠긴 것으로 보고 픽셀이 많은
        쪽만 단일 라인으로 강등한다(허위 양쪽 검출 배제).
      - prior_half_px: 한쪽 소실 폴백에서 쓸 반차폭(BEV px). 노드가 직전 양쪽
        검출에서 실측한 차폭을 EMA 로 기억해 넘긴다. None/0 이면 lane_width_ratio
        기반 고정 추정으로 폴백. 실측 차폭을 쓰면 고정 추정 오차로 인한 편향
        (한쪽 치우침)이 사라진다. 반환 LaneResult.lane_width_px 로 이번 프레임의
        반차폭을 되돌려 노드가 다음 프레임 prior_half_px 로 되먹인다.
    """
    if image_bgr is None or image_bgr.size == 0:
        return LaneResult(0.0, False, 0.0)

    h, w = image_bgr.shape[:2]

    # 1) 이진화
    if method == METHOD_COLOR:
        mask = _color_mask(image_bgr, hsv_lower, hsv_upper)
    elif method == METHOD_WHITE:
        mask = _white_mask(image_bgr, white_s_max, white_v_min, blur_ksize)
        # 견고성 옵션: 흰색(채도 게이팅) 마스크와 명암 마스크를 AND 로 결합.
        if white_combine:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            mask = cv2.bitwise_and(
                mask, _binarize_lane(gray, polarity, block_size, blur_ksize))
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mask = _binarize_lane(gray, polarity, block_size, blur_ksize)
    mask = _denoise(mask, morph_ksize)
    mask_px = int((mask > 0).sum())

    # 2) BEV 변환
    M = build_perspective(w, h, bev_src_tl, bev_src_tr, bev_src_br, bev_src_bl,
                          warp_w, warp_h)
    binary_bev = cv2.warpPerspective(mask, M, (warp_w, warp_h),
                                     flags=cv2.INTER_NEAREST)

    total_px = int((binary_bev > 0).sum())
    overlay = cv2.cvtColor(binary_bev, cv2.COLOR_GRAY2BGR) if draw else None

    if total_px < valid_min_px:
        return LaneResult(0.0, False, 0.0, pixels=total_px, mask_pixels=mask_px,
                          valid_bands=0, overlay=overlay)

    # 3~4) 히스토그램 + 슬라이딩 윈도우
    leftx_base, rightx_base = find_lane_bases(binary_bev, hist_ratio)
    nz = binary_bev.nonzero()
    nonzero_y = np.array(nz[0])
    nonzero_x = np.array(nz[1])
    left_inds = sliding_window_search(binary_bev, leftx_base, nonzero_x, nonzero_y,
                                      n_windows, margin, minpix, overlay, (0, 255, 0))
    right_inds = sliding_window_search(binary_bev, rightx_base, nonzero_x, nonzero_y,
                                       n_windows, margin, minpix, overlay, (0, 128, 255))

    # 5) 2차 함수 적합
    left_fit = right_fit = None
    if len(left_inds) >= min_lane_px:
        left_fit = np.polyfit(nonzero_y[left_inds], nonzero_x[left_inds], 2)
    if len(right_inds) >= min_lane_px:
        right_fit = np.polyfit(nonzero_y[right_inds], nonzero_x[right_inds], 2)

    left_detected = left_fit is not None
    right_detected = right_fit is not None

    bev_h, bev_w = binary_bev.shape[:2]
    y_near = bev_h - 1
    y_far = 0
    img_half = bev_w / 2.0

    # 허위 양쪽 검출 배제: 둘 다 적합됐어도 near 간격이 너무 좁으면(혹은 교차하면)
    # 두 슬라이딩 윈도우가 같은 물리 라인에 겹쳐 잠긴 것 → 픽셀 많은 쪽만 남긴다.
    # 겹친 상태로 평균을 내면 중앙이 그 한 선 위로 끌려가 한쪽만 추종하게 된다.
    if left_detected and right_detected:
        sep_near = _poly_x(right_fit, y_near) - _poly_x(left_fit, y_near)
        if sep_near < min_sep_ratio * bev_w:
            if len(left_inds) >= len(right_inds):
                right_detected, right_fit = False, None
            else:
                left_detected, left_fit = False, None

    # 한쪽 소실 폴백 반차폭: 메모리(prior_half_px, 직전 실측 차폭)를 우선 쓰고,
    # 없으면 lane_width_ratio 기반 고정 추정. 고정 추정은 실제 차폭과 어긋나면
    # 남은 한 선 기준 중앙이 편향돼 "한쪽만 보고 달리는" 증상을 만든다.
    fallback_half = (float(prior_half_px)
                     if prior_half_px and float(prior_half_px) > 0.0
                     else lane_width_ratio * bev_w / 2.0)

    def lane_center_at(y):
        if left_detected and right_detected:
            return (_poly_x(left_fit, y) + _poly_x(right_fit, y)) / 2.0
        if left_detected:
            return _poly_x(left_fit, y) + fallback_half
        if right_detected:
            return _poly_x(right_fit, y) - fallback_half
        return None

    center_near = lane_center_at(y_near)
    center_far = lane_center_at(y_far)

    if center_near is None:
        return LaneResult(0.0, False, 0.0, pixels=total_px, mask_pixels=mask_px,
                          valid_bands=0, overlay=overlay)

    valid_bands = 2 if (left_detected and right_detected) else 1

    # 이번 프레임 반차폭: 양쪽 검출 시 실측(near), 아니면 폴백값. 노드가 EMA 로
    # 기억해 다음 프레임 prior_half_px 로 되먹인다(한쪽 소실 구간 중앙 유지).
    if left_detected and right_detected:
        lane_width_px = max(
            0.0, (_poly_x(right_fit, y_near) - _poly_x(left_fit, y_near)) / 2.0)
    else:
        lane_width_px = fallback_half

    def norm(cx):
        return float(np.clip((cx - img_half) / img_half, -1.0, 1.0))

    off_near = norm(center_near)
    off_far = norm(center_far) if center_far is not None else off_near
    # near 가중 + far 피드포워드(기존 규약 유지).
    offset = float(np.clip((2.0 * off_near + off_far) / 3.0, -1.0, 1.0))
    curvature = float(np.clip((off_far - off_near) / 2.0, -1.0, 1.0))

    # 디버그 오버레이: 적합곡선 + 중앙선 표시
    if draw and overlay is not None:
        ploty = np.linspace(0, bev_h - 1, bev_h)
        for fit, col in ((left_fit, (0, 255, 0)), (right_fit, (0, 128, 255))):
            if fit is not None:
                fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
                pts = np.int32(np.stack([np.clip(fitx, 0, bev_w - 1), ploty], axis=1))
                cv2.polylines(overlay, [pts], False, col, 2)
        cv2.line(overlay, (int(img_half), 0), (int(img_half), bev_h - 1),
                 (120, 120, 120), 1)
        cv2.circle(overlay, (int(np.clip(center_near, 0, bev_w - 1)), y_near - 2),
                   4, (0, 0, 255), -1)

    return LaneResult(offset, True, curvature, pixels=total_px,
                      mask_pixels=mask_px, valid_bands=valid_bands,
                      left_detected=left_detected,
                      right_detected=right_detected,
                      lane_width_px=float(lane_width_px), overlay=overlay)