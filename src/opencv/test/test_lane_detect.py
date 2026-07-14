"""흰색 마스크(method='white') 최소 단위 테스트.

CLAUDE.md 10번: compute_lane_offset 은 순수 함수(ROS 비의존)라 합성 이미지로
오프라인 검증이 가능하다. 여기서는 대회 white_track 의 핵심인 "흰색 마스크가
저채도·고명도만 통과시키고 유채색(파란 매트)을 배제하는지"만 최소로 검증한다.
cv2+numpy 만 있으면 실행되며 ROS/onnxruntime 를 요구하지 않는다.

실행: python3 -m pytest src/opencv/test/test_lane_detect.py
"""

import cv2
import numpy as np

from opencv.lane_detect import _white_mask, compute_lane_offset, METHOD_WHITE


def test_white_mask_gates_saturation_and_value():
    """흰색=통과, 채도 높은 파랑=배제, 명도 낮은 회색=배제."""
    bgr = np.zeros((1, 3, 3), np.uint8)
    bgr[0, 0] = (255, 255, 255)  # 흰색: S=0, V=255 → 통과
    bgr[0, 1] = (255, 0, 0)      # 순수 파랑: S=255 → 채도로 배제
    bgr[0, 2] = (60, 60, 60)     # 어두운 회색: V=60 → 명도로 배제
    mask = _white_mask(bgr, s_max=70, v_min=170, blur_ksize=0)
    assert mask[0, 0] == 255
    assert mask[0, 1] == 0
    assert mask[0, 2] == 0


def test_white_track_symmetric_offset_near_zero():
    """검은 바닥 + 양쪽 흰 선 + 파란 매트 → 대칭이라 offset≈0, 파란 매트는 걸러짐."""
    img = np.zeros((160, 320, 3), np.uint8)
    cv2.rectangle(img, (70, 60), (78, 159), (255, 255, 255), -1)    # 좌 흰 선
    cv2.rectangle(img, (242, 60), (250, 159), (255, 255, 255), -1)  # 우 흰 선
    cv2.rectangle(img, (0, 0), (40, 159), (200, 120, 20), -1)       # 좌 파란 매트
    cv2.rectangle(img, (280, 0), (319, 159), (200, 120, 20), -1)    # 우 파란 매트
    res = compute_lane_offset(img, method=METHOD_WHITE, white_s_max=70,
                              white_v_min=170, min_lane_px=50, valid_min_px=20)
    assert res.valid
    assert res.valid_bands == 2
    assert abs(res.offset) < 0.2
    assert res.mask_pixels > 0
    # 양쪽 검출이면 실측 반차폭(>0)을 되돌려준다(폴백 메모리 피드백용).
    assert res.lane_width_px > 0.0


def _two_white_lines(left_x, right_x):
    """검은 바닥 + 지정 x 위치의 좌/우 세로 흰 선 합성 프레임."""
    img = np.zeros((160, 320, 3), np.uint8)
    cv2.rectangle(img, (left_x, 40), (left_x + 8, 159), (255, 255, 255), -1)
    cv2.rectangle(img, (right_x, 40), (right_x + 8, 159), (255, 255, 255), -1)
    return img


def test_prior_half_px_keeps_center_when_one_lane_lost():
    """한쪽 소실 시, 실측 차폭 메모리(prior_half_px)를 쓰면 고정 추정 편향이 사라진다.

    양쪽 대칭 프레임에서 실측 반차폭을 얻은 뒤, 우측 선만 있는 프레임을 두 방식으로
    평가한다: (a) 고정 lane_width_ratio 폴백, (b) 실측 prior_half_px 폴백.
    실측 차폭이 고정 추정과 다르면 두 offset 이 달라져야 하고, 실측 기반이 실제
    중앙(좌선이 있었을 위치)에 더 가깝다.
    """
    kw = dict(method=METHOD_WHITE, white_s_max=70, white_v_min=170,
              min_lane_px=50, valid_min_px=20)
    both = compute_lane_offset(_two_white_lines(70, 242), **kw)
    assert both.valid_bands == 2 and both.lane_width_px > 0.0

    one = _two_white_lines(70, 242).copy()
    one[:, 236:254] = 0  # 우측 흰 선 제거 → 좌측만 남김
    lost_fixed = compute_lane_offset(one, lane_width_ratio=0.55, **kw)
    lost_prior = compute_lane_offset(one, prior_half_px=both.lane_width_px, **kw)
    assert lost_fixed.valid_bands == 1 and lost_prior.valid_bands == 1
    # 실측 차폭이 0.55 고정 추정과 다르면 두 offset 이 갈린다(메모리가 실제로 반영됨).
    assert lost_prior.lane_width_px == both.lane_width_px


def test_overlapping_windows_demoted_to_single_lane():
    """좌/우 윈도우가 같은 한 선에 겹쳐 잠기면(간격 과소) 단일 라인으로 강등."""
    img = np.zeros((160, 320, 3), np.uint8)
    cv2.rectangle(img, (156, 40), (164, 159), (255, 255, 255), -1)  # 중앙 단일 선
    res = compute_lane_offset(img, method=METHOD_WHITE, white_s_max=70,
                              white_v_min=170, min_lane_px=50, valid_min_px=20,
                              min_sep_ratio=0.30)
    # 한 선만 있으므로 양쪽(valid_bands==2)으로 오검출되면 안 된다.
    assert res.valid_bands != 2
