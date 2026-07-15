"""차선 검출·조향 오프라인 튜닝 하니스 (실차 없이 rosbag/영상/이미지로 검증).

BEV(조감도)+슬라이딩 윈도우 재작성(opencv/lane_detect.py) 이후 버전.
**배포되는 실제 프로덕션 모듈**을 그대로 호출한다:
    - 검출: opencv.lane_detect.compute_lane_offset  (opencv_node 가 쓰는 그 함수)
    - 제어: inference.driving_policy.DrivingPolicy   (inference_node 가 쓰는 그 정책)

파라미터 이름·의미가 opencv_node/inference_node 의 ROS param 과 1:1 대응하므로,
좋은 값을 찾으면 그대로 auto_driving.launch.py / opencv_node param 으로 옮기면 된다.
opencv_node 가 하는 한쪽-소실 폴백 반차폭 EMA 되먹임까지 복제해, 프레임 시퀀스
(영상/bag)에서 실차 노드와 동일한 offset/curvature 를 재현한다.

입력 3종 (확장자/경로로 자동 판별):
    - rosbag2      : 디렉터리(예: bagfile/track_full_.../) 또는 그 안의 *.db3
                     → /camera/image/compressed(JPEG) 프레임을 디코드해 재생.
                     ⚠️ ROS 환경 필요: `source /opt/ros/humble/setup.bash` 선행.
    - 영상 파일    : *.mp4 / *.avi …
    - 이미지 파일  : *.jpg / *.png …

산출: 원본 프레임(중심선·추정 차선 중앙·조향 바·수치)과 BEV 오버레이(슬라이딩
윈도우/적합곡선/중앙점)를 좌우로 붙인 오버레이 영상/이미지. BEV 4점 캘리브레이션은
이 오버레이의 조감도가 "직선 구간에서 좌우 차선이 세로 평행선"이 되도록 맞춘다.

사용법:
    # rosbag 리플레이(권장 — 실차 카메라 그대로)
    source /opt/ros/humble/setup.bash
    python3 tools/lane_tuning_harness.py bagfile/track_full_20260714_082403 out.mp4

    # 영상/이미지
    python3 tools/lane_tuning_harness.py capture.mp4
    python3 tools/lane_tuning_harness.py frame.jpg --profile white_track

    # 파라미터 오버라이드(opencv_node ROS param 과 동일 이름·의미)
    python3 tools/lane_tuning_harness.py <bag> --white-s-max 40 --white-v-min 170 \
        --bev-tl 0.22 0.60 --bev-tr 0.78 0.60 --bev-br 1.02 1.00 --bev-bl -0.02 1.00

권장 절차:
    1) 실트랙에서 tools/record_track.sh 로 bag 녹화(또는 이미 있는 bag 사용).
    2) 이 하니스로 오버레이 영상을 만들어 흰 차선이 BEV 에서 잘 잡히는지,
       offset/curvature 부호·crop 이 맞는지 눈으로 확인한다.
    3) 잘 잡히는 param 값을 auto_driving.launch.py / opencv_node ROS param 으로 이관.
    4) 실차에서 `ros2 topic info /lane/offset` 로 pub/sub 연결·조향 방향 재확인.

주의: 이 하니스는 흰/색 차선 추종만 본다. 신호등(B.1/B.6)·좌우표지판(B.3)·
ArUco(B.4) 는 YOLO/aruco 별도 경로 몫이라 다루지 않는다(require_green_start=False
로 강제해 차선 거동만 관찰).
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

# opencv_node.py 와 동일한 프로파일 프리셋(method/polarity).
PROFILES = {
    'white_track': {'method': 'white', 'polarity': 'light'},   # 대회(기본)
    'orange_track': {'method': 'color', 'polarity': 'dark'},   # 연습
}

CAMERA_TOPIC = '/camera/image/compressed'


# --------------------------------------------------------------------------- #
# 인자
# --------------------------------------------------------------------------- #
def build_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', help='rosbag 디렉터리/*.db3, 영상(.mp4), 또는 이미지(.jpg)')
    ap.add_argument('output', nargs='?', default=None,
                    help='오버레이 출력(생략 시 lane_overlay.<mp4|jpg>)')
    ap.add_argument('--profile', choices=list(PROFILES), default='white_track')
    ap.add_argument('--topic', default=CAMERA_TOPIC, help='bag 카메라 토픽')
    ap.add_argument('--stride', type=int, default=1, help='N 프레임마다 1장 처리')
    ap.add_argument('--max-frames', type=int, default=0, help='처리 최대 프레임(0=전체)')
    # --- 프리셋 덮어쓰기(명시 시에만) — opencv_node 와 동일 규약 ---
    ap.add_argument('--method', choices=['white', 'brightness', 'color'], default=None)
    ap.add_argument('--polarity', choices=['light', 'dark'], default=None)
    # --- 검출 param (opencv_node ROS param 과 1:1) ---
    ap.add_argument('--white-s-max', type=int, default=50)
    ap.add_argument('--white-v-min', type=int, default=150)
    ap.add_argument('--white-combine', action='store_true')
    ap.add_argument('--block-size', type=int, default=25)
    ap.add_argument('--blur-ksize', type=int, default=5)
    ap.add_argument('--morph-ksize', type=int, default=3)
    ap.add_argument('--valid-min-px', type=int, default=40)
    ap.add_argument('--hsv-lower', type=int, nargs=3, default=[5, 80, 80])
    ap.add_argument('--hsv-upper', type=int, nargs=3, default=[22, 255, 255])
    # BEV 4점(원본 폭/높이 대비 0~1 비율) — 기본값은 opencv_node 기본과 동일
    # (2026-07-15 캘리브레이션, bag_20260715_122828 640x480 white_track).
    ap.add_argument('--bev-tl', type=float, nargs=2, default=[0.25, 0.50])
    ap.add_argument('--bev-tr', type=float, nargs=2, default=[0.86, 0.50])
    ap.add_argument('--bev-br', type=float, nargs=2, default=[1.15, 1.00])
    ap.add_argument('--bev-bl', type=float, nargs=2, default=[-0.05, 1.00])
    ap.add_argument('--warp-w', type=int, default=200)
    ap.add_argument('--warp-h', type=int, default=240)
    ap.add_argument('--n-windows', type=int, default=10)
    ap.add_argument('--margin', type=int, default=30)
    ap.add_argument('--minpix', type=int, default=25)
    ap.add_argument('--hist-ratio', type=float, default=0.5)
    ap.add_argument('--min-lane-px', type=int, default=200)
    ap.add_argument('--lane-width-ratio', type=float, default=0.55)
    ap.add_argument('--min-sep-ratio', type=float, default=0.30)
    ap.add_argument('--horiz-filter-frac', type=float, default=0.35)
    ap.add_argument('--horiz-filter-pad', type=int, default=6)
    ap.add_argument('--lane-width-ema', type=float, default=0.3)
    # --- 제어 param (inference_node ROS param 과 1:1) ---
    ap.add_argument('--curve-ff', type=float, default=default_params()['curve_ff'])
    ap.add_argument('--steer-kp', type=float, default=default_params()['steer_kp'])
    ap.add_argument('--steer-kd', type=float, default=default_params()['steer_kd'])
    ap.add_argument('--steer-sign', type=float, default=default_params()['steer_sign'])
    return ap.parse_args()


def resolve_detect_kwargs(a):
    """CLI → compute_lane_offset kwargs (현재 BEV 시그니처). opencv_node 매핑과 동일."""
    preset = PROFILES[a.profile]
    method = a.method if a.method else preset['method']
    polarity = a.polarity if a.polarity else preset['polarity']
    return dict(
        method=method, polarity=polarity,
        hsv_lower=tuple(a.hsv_lower), hsv_upper=tuple(a.hsv_upper),
        white_s_max=a.white_s_max, white_v_min=a.white_v_min,
        white_combine=a.white_combine,
        block_size=a.block_size, blur_ksize=a.blur_ksize, morph_ksize=a.morph_ksize,
        bev_src_tl=tuple(a.bev_tl), bev_src_tr=tuple(a.bev_tr),
        bev_src_br=tuple(a.bev_br), bev_src_bl=tuple(a.bev_bl),
        warp_w=a.warp_w, warp_h=a.warp_h,
        n_windows=a.n_windows, margin=a.margin, minpix=a.minpix,
        hist_ratio=a.hist_ratio, min_lane_px=a.min_lane_px,
        lane_width_ratio=a.lane_width_ratio, min_sep_ratio=a.min_sep_ratio,
        horiz_filter_frac=a.horiz_filter_frac, horiz_filter_pad=a.horiz_filter_pad,
        valid_min_px=a.valid_min_px, draw=True,   # draw=True → BEV 오버레이 확보
    ), method, polarity


def make_policy(a):
    # 차선 거동만 관찰: 출발 게이트/미션 로직은 끈다.
    return DrivingPolicy({
        'require_green_start': False,
        'curve_ff': a.curve_ff, 'steer_kp': a.steer_kp,
        'steer_kd': a.steer_kd, 'steer_sign': a.steer_sign,
    })


# --------------------------------------------------------------------------- #
# 프레임 처리 — opencv_node 의 EMA 반차폭 되먹임을 복제(상태 보존)
# --------------------------------------------------------------------------- #
class LaneRunner:
    """프레임을 받아 compute_lane_offset + DrivingPolicy 를 돌린다.

    opencv_node.image_callback 의 한쪽-소실 폴백 반차폭 메모리(EMA)를 그대로
    복제해, 영상/bag 재생 시 실차 노드와 동일한 offset/curvature 를 재현한다.
    """

    def __init__(self, detect_kwargs, policy, lane_width_ema):
        self.detect_kwargs = detect_kwargs
        self.policy = policy
        self.lane_width_ema = lane_width_ema
        self._lane_half_px = 0.0

    def step(self, frame):
        prior = self._lane_half_px if self._lane_half_px > 0.0 else None
        lane = compute_lane_offset(frame, prior_half_px=prior, **self.detect_kwargs)
        # opencv_node 와 동일: 양쪽 검출(valid_bands==2) 프레임만 EMA 갱신.
        if (lane.valid and lane.valid_bands == 2 and lane.lane_width_px > 0.0
                and self.lane_width_ema > 0.0):
            if self._lane_half_px <= 0.0:
                self._lane_half_px = lane.lane_width_px
            else:
                a = self.lane_width_ema
                self._lane_half_px = (1.0 - a) * self._lane_half_px + a * lane.lane_width_px
        steer, throttle = self.policy.step(
            LaneSignal(offset=lane.offset, valid=lane.valid, curvature=lane.curvature))
        return lane, steer, throttle


def compose_overlay(frame, lane, steer, throttle):
    """원본 주석 프레임 | BEV 오버레이 를 좌우로 결합해 반환."""
    out = frame.copy()
    h, w = out.shape[:2]
    cx = w // 2
    cv2.line(out, (cx, 0), (cx, h), (255, 128, 0), 1)          # 이미지 중심선(파랑)
    if lane.valid:
        lane_cx = int(cx + lane.offset * (w / 2.0))
        cv2.line(out, (lane_cx, 0), (lane_cx, h), (0, 255, 0), 2)  # 추정 차선 중앙(초록)
    bar_y = h - 18
    cv2.line(out, (cx, bar_y), (int(cx + steer * (w / 3.0)), bar_y), (0, 0, 255), 6)
    txt = (f'off={lane.offset:+.2f} cur={lane.curvature:+.2f} '
           f'valid={int(lane.valid)} bands={lane.valid_bands} '
           f'steer={steer:+.2f} thr={throttle:.2f}')
    cv2.putText(out, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    # BEV 오버레이(있으면)를 원본 높이에 맞춰 오른쪽에 붙인다.
    if lane.overlay is not None:
        bev = lane.overlay
        scale = h / bev.shape[0]
        bev = cv2.resize(bev, (int(bev.shape[1] * scale), h),
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(bev, 'BEV', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        out = np.hstack([out, bev])
    return out


# --------------------------------------------------------------------------- #
# 입력 소스
# --------------------------------------------------------------------------- #
def is_image(path):
    return path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))


def is_bag(path):
    return os.path.isdir(path) or path.lower().endswith('.db3')


def iter_bag_frames(path, topic):
    """rosbag2(디렉터리 또는 *.db3)에서 CompressedImage 프레임(BGR)을 순서대로 yield."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f'[bag] ROS 파이썬 모듈 import 실패({e}). '
            '`source /opt/ros/humble/setup.bash` 후 다시 실행하세요.')
    uri = os.path.dirname(path) if path.lower().endswith('.db3') else path
    uri = uri.rstrip('/') or '.'
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in types:
        raise SystemExit(f'[bag] 토픽 {topic} 없음. 존재 토픽: {sorted(types)}')
    MsgT = get_message(types[topic])
    while reader.has_next():
        tname, data, _ = reader.read_next()
        if tname != topic:
            continue
        msg = deserialize_message(data, MsgT)
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            yield img


def iter_video_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# 실행
# --------------------------------------------------------------------------- #
def run_stream(a, frames, runner, out_path):
    """프레임 스트림 → 오버레이 mp4. 첫 프레임에서 writer 크기를 확정한다."""
    writer = None
    n = valid = kept = 0
    for i, frame in enumerate(frames):
        if a.stride > 1 and i % a.stride != 0:
            continue
        lane, steer, throttle = runner.step(frame)
        canvas = compose_overlay(frame, lane, steer, throttle)
        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     20.0, (w, h))
        writer.write(canvas)
        kept += 1
        valid += int(lane.valid)
        if a.max_frames and kept >= a.max_frames:
            break
    if writer is not None:
        writer.release()
    print(f'[stream] {kept} 프레임 처리, valid차선 {valid} '
          f'({100 * valid / max(1, kept):.0f}%) -> {out_path}')


def run_image(a, runner, out_path):
    frame = cv2.imread(a.source)
    if frame is None:
        raise FileNotFoundError(a.source)
    lane, steer, throttle = runner.step(frame)
    cv2.imwrite(out_path, compose_overlay(frame, lane, steer, throttle))
    print(f'[image] valid={lane.valid} offset={lane.offset:+.3f} '
          f'curvature={lane.curvature:+.3f} pixels={lane.pixels} '
          f'valid_bands={lane.valid_bands} steer={steer:+.3f} throttle={throttle:.3f}')
    print(f'        overlay -> {out_path}')


def main():
    a = build_args()
    detect_kwargs, method, polarity = resolve_detect_kwargs(a)
    runner = LaneRunner(detect_kwargs, make_policy(a), a.lane_width_ema)

    src_is_image = is_image(a.source)
    if a.output:
        out_path = a.output
    else:
        out_path = 'lane_overlay.jpg' if src_is_image else 'lane_overlay.mp4'

    print(f'source={a.source} profile={a.profile} method={method} polarity={polarity} '
          f'| white_s_max={a.white_s_max} white_v_min={a.white_v_min} '
          f'curve_ff={a.curve_ff} steer_kp={a.steer_kp} steer_sign={a.steer_sign}')

    if src_is_image:
        run_image(a, runner, out_path)
    elif is_bag(a.source):
        run_stream(a, iter_bag_frames(a.source, a.topic), runner, out_path)
    else:
        run_stream(a, iter_video_frames(a.source), runner, out_path)


if __name__ == '__main__':
    main()
