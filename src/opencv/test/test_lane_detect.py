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
