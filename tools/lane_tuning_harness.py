"""차선 검출·조향 오프라인 튜닝 하니스 (실차 없이 이미지/영상으로 검증).

sim_line_260707_fix.py 의 "테스트 하니스" 아이디어를 이식하되, **배포되는 실제
프로덕션 모듈**을 그대로 호출한다:
    - 검출: opencv.lane_detect.compute_lane_offset  (opencv_node 가 쓰는 그 함수)
    - 제어: inference.driving_policy.DrivingPolicy   (inference_node 가 쓰는 그 정책)

따라서 여기서 튜닝한 파라미터(--roi-top, --block-size, --valid-min-px, --curve-ff …)는
이름·의미가 ROS param 과 1:1 대응하므로, 좋은 값을 찾으면 그대로
auto_driving.launch.py / opencv_node 에 옮기면 된다(재현성 확보).

프로파일 프리셋은 opencv_node.py 와 동일:
    white_track (기본, 대회): brightness / light / split=True
    orange_track (연습):      color / dark / split=False

사용법:
    python3 tools/lane_tuning_harness.py <이미지|영상> [출력]
        [--profile white_track|orange_track]
        [--roi-top 50] [--num-bands 3] [--valid-min-px 40]
        [--block-size 25] [--morph-ksize 3] [--lane-half-norm 0.5]
        [--curve-ff 0.30] [--steer-kp 0.6] [--steer-kd 0.15]

권장 절차:
    1) 보드에서 `ros2 topic echo /camera/image/compressed` 대신, 카메라 프레임을
       몇 장/짧은 영상으로 저장해 이 개발 박스로 가져온다.
    2) 이 하니스로 오버레이를 만들어 흰 차선이 초록선(추정 차선 중앙)으로 잘
       잡히는지, offset/curvature 부호가 맞는지 눈으로 확인한다.
    3) 잘 잡히는 파라미터를 찾으면 그 값을 launch/opencv_node ROS param 으로 이관.
    4) 실차에서 `ros2 topic info /lane/offset` 로 pub/sub 연결과 조향 방향 재확인.

주의: 이 하니스는 흰/색 차선 추종만 본다. 신호등(B.1/B.6)·좌우표지판(B.3)·
ArUco(B.4) 는 YOLO/aruco 별도 경로 몫이라 여기서는 다루지 않는다(초록불 게이트가
없도록 require_green_start=False 로 강제해 차선 거동만 관찰한다).
"""
import argparse
import os
import sys

import cv2
import numpy as np

# --- 실제 배포 모듈 경로 등록(패키지 소스 트리 직접 import) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, '..', 'src')
sys.path.insert(0, os.path.join(_SRC, 'opencv'))      # -> opencv.lane_detect
sys.path.insert(0, os.path.join(_SRC, 'inference'))   # -> inference.driving_policy

from opencv.lane_detect import compute_lane_offset  # noqa: E402
from inference.driving_policy import (  # noqa: E402
    DrivingPolicy, LaneSignal, default_params,
)

# opencv_node.py 와 동일한 프로파일 프리셋.
PROFILES = {
    'white_track': {'method': 'brightness', 'polarity': 'light', 'split': True},
    'orange_track': {'method': 'color', 'polarity': 'dark', 'split': False},
}


def build_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', help='입력 이미지(.jpg/.png…) 또는 영상(.mp4…)')
    ap.add_argument('output', nargs='?', default=None,
                    help='오버레이 출력 경로(생략 시 lane_overlay.<입력확장자>)')
    ap.add_argument('--profile', choices=list(PROFILES), default='white_track')
    # --- 검출 param (opencv_node ROS param 과 1:1) ---
    ap.add_argument('--roi-top', type=int, default=50)
    ap.add_argument('--roi-left', type=int, default=0)
    ap.add_argument('--roi-right', type=int, default=-1, help='-1 = 전폭')
    ap.add_argument('--num-bands', type=int, default=3)
    ap.add_argument('--valid-min-px', type=int, default=40)
    ap.add_argument('--block-size', type=int, default=25)
    ap.add_argument('--morph-ksize', type=int, default=3)
    ap.add_argument('--lane-half-norm', type=float, default=0.5)
    # 프로파일 프리셋을 덮어쓰고 싶을 때만 지정(opencv_node 와 동일 규약).
    ap.add_argument('--method', choices=['brightness', 'color'], default=None)
    ap.add_argument('--polarity', choices=['light', 'dark'], default=None)
    ap.add_argument('--split-lanes', choices=['auto', 'true', 'false'], default='auto')
    # --- 제어 param (inference_node ROS param 과 1:1) ---
    ap.add_argument('--curve-ff', type=float, default=default_params()['curve_ff'])
    ap.add_argument('--steer-kp', type=float, default=default_params()['steer_kp'])
    ap.add_argument('--steer-kd', type=float, default=default_params()['steer_kd'])
    ap.add_argument('--steer-sign', type=float, default=default_params()['steer_sign'])
    return ap.parse_args()


def resolve_detect_kwargs(a):
    preset = PROFILES[a.profile]
    method = a.method if a.method else preset['method']
    polarity = a.polarity if a.polarity else preset['polarity']
    if a.split_lanes == 'true':
        split = True
    elif a.split_lanes == 'false':
        split = False
    else:
        split = preset['split']
    roi_right = None if a.roi_right < 0 else a.roi_right
    return dict(
        roi_top=a.roi_top, roi_left=a.roi_left, roi_right=roi_right,
        num_bands=a.num_bands, valid_min_px=a.valid_min_px,
        polarity=polarity, method=method, morph_ksize=a.morph_ksize,
        split_lanes=split, lane_half_norm=a.lane_half_norm,
        block_size=a.block_size,
    ), method, polarity, split


def make_policy(a):
    # 차선 거동만 관찰: 출발 게이트/미션 로직은 끈다.
    return DrivingPolicy({
        'require_green_start': False,
        'curve_ff': a.curve_ff, 'steer_kp': a.steer_kp,
        'steer_kd': a.steer_kd, 'steer_sign': a.steer_sign,
    })


def draw_overlay(frame, lane, steer, throttle, roi_top):
    out = frame.copy()
    h, w = out.shape[:2]
    cx = w // 2
    # 이미지 중심선(파랑).
    cv2.line(out, (cx, roi_top), (cx, h), (255, 128, 0), 1)
    # 추정 차선 중앙(초록) — offset 은 원본 이미지 중앙 기준 정규화.
    if lane.valid:
        lane_cx = int(cx + lane.offset * (w / 2.0))
        cv2.line(out, (lane_cx, roi_top), (lane_cx, h), (0, 255, 0), 2)
    # 조향 바(빨강).
    bar_y = h - 18
    cv2.line(out, (cx, bar_y), (int(cx + steer * (w / 3.0)), bar_y), (0, 0, 255), 6)
    txt = (f'off={lane.offset:+.2f} cur={lane.curvature:+.2f} '
           f'valid={int(lane.valid)} bands={lane.valid_bands} '
           f'steer={steer:+.2f} thr={throttle:.2f}')
    cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def process(frame, detect_kwargs, policy):
    lane = compute_lane_offset(frame, **detect_kwargs)
    steer, throttle = policy.step(
        LaneSignal(offset=lane.offset, valid=lane.valid, curvature=lane.curvature))
    return lane, steer, throttle


def run_image(a, detect_kwargs, policy, out_path):
    frame = cv2.imread(a.source)
    if frame is None:
        raise FileNotFoundError(a.source)
    lane, steer, throttle = process(frame, detect_kwargs, policy)
    cv2.imwrite(out_path, draw_overlay(frame, lane, steer, throttle, a.roi_top))
    print(f'[image] valid={lane.valid} offset={lane.offset:+.3f} '
          f'curvature={lane.curvature:+.3f} pixels={lane.pixels} '
          f'valid_bands={lane.valid_bands} steer={steer:+.3f} throttle={throttle:.3f}')
    print(f'        overlay -> {out_path}')


def run_video(a, detect_kwargs, policy, out_path):
    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise FileNotFoundError(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    n = valid = 0
    for _ in iter(int, 1):
        ok, frame = cap.read()
        if not ok:
            break
        lane, steer, throttle = process(frame, detect_kwargs, policy)
        writer.write(draw_overlay(frame, lane, steer, throttle, a.roi_top))
        n += 1
        valid += int(lane.valid)
    cap.release()
    writer.release()
    print(f'[video] {n} frames, valid차선 {valid} ({100*valid/max(1,n):.0f}%) -> {out_path}')


def main():
    a = build_args()
    detect_kwargs, method, polarity, split = resolve_detect_kwargs(a)
    policy = make_policy(a)

    is_image = a.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    if a.output:
        out_path = a.output
    else:
        ext = 'jpg' if is_image else 'mp4'
        out_path = f'lane_overlay.{ext}'

    print(f'profile={a.profile} method={method} polarity={polarity} split={split} '
          f'| curve_ff={a.curve_ff} steer_kp={a.steer_kp} steer_kd={a.steer_kd}')
    if is_image:
        run_image(a, detect_kwargs, policy, out_path)
    else:
        run_video(a, detect_kwargs, policy, out_path)


if __name__ == '__main__':
    main()
