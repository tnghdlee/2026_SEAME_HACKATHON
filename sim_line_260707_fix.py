"""
white_lane_follower.py
D-Racer-Kit  흰색 차선(white_track) 추종 자율주행 프로토타입

목표: 검은 바닥 + 양쪽 흰 경계선 트랙을 "차선을 벗어나지 않고" 완주하도록 조향/스로틀을 산출한다.

이전 노트북 코드(grayscale -> blur -> ROI) 계승 + 아래 3가지를 추가:
  1) 흰색 선 전용 이진화 (어두운 바닥 위 밝은 선 -> adaptive threshold)
  2) 곡선 대응: 밴드 무게중심(band-centroid) + 곡률(curvature) 피드포워드
  3) 이탈 방지: 차선 로스트 시 "마지막 조향 유지" + 감속 폴백

출력: offset[-1..1], curvature[-1..1], steering[-1..1], throttle[0..1]
CLAUDE.md 10.2 기본값과 정합 (steer_kp=0.6, steer_kd=0.15, cruise_throttle=0.13, curve_slow=0.5 ...)

주의:
  - 이 개발 박스엔 카메라/실차가 없으므로 로직·산술만 작성했다. 임계값(adaptive C, ROI, valid_min_px 등)은
    반드시 보드에서 /opencv/image/edge 와 진단 로그를 보며 실측 튜닝할 것(CLAUDE.md 10번 조명 민감성 경고).
  - 실차 통합 경로는 파일 하단 "PROJECT INTEGRATION" 주석 참조. 통합 시 8.2(/lane/offset 절대표기) 먼저 확인.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class LaneConfig:
    # ROI: 하단만 사용(하늘/배경 제거). roi_top_ratio=0.5 -> 아래 절반만 본다.
    roi_top_ratio: float = 0.5
    roi_left: int = 0            # 좌측 크롭(px). 비대칭 ROI 필요 시 사용
    roi_right: int = -1          # 우측 크롭(px). -1 = 전폭

    # 전처리 (노트북 계승)
    blur_ksize: int = 5          # 홀수

    # 흰색 선 이진화 (white_track: 어두운 바닥 + 밝은 선)
    use_adaptive: bool = True    # True: 조명 불균일에 강한 adaptive / False: 고정 임계값
    adaptive_block: int = 25     # 홀수. 지역 평균 창 크기
    adaptive_C: int = -10        # 음수 -> "지역 평균보다 밝은" 픽셀만 선으로 (밝은 흰선 추출)
    fixed_thresh: int = 180      # use_adaptive=False 일 때 밝기 임계값
    morph_ksize: int = 3         # 형태학적 열림 커널(작은 노이즈 제거). <=1 이면 비활성

    # 밴드 무게중심
    num_bands: int = 3           # 하단 ROI를 수평으로 몇 개 밴드로 나눌지
    split_lanes: bool = True     # True: 좌/우 선을 나눠 중점을 차선 중앙으로 (white_track 기본)
    valid_min_px: int = 40       # split_lanes=False 시 밴드 유효 최소 픽셀
    side_min_px: int = 20        # split_lanes=True 시 한쪽(좌 or 우) 유효 최소 픽셀
    lane_half_norm: float = 0.5  # 한쪽 선만 보일 때 반대편을 이 비율(정규화)만큼 떨어졌다고 가정

    def __post_init__(self):
        # OpenCV 제약: blur/adaptive_block 는 홀수여야 함
        if self.blur_ksize % 2 == 0:
            self.blur_ksize += 1
        if self.adaptive_block % 2 == 0:
            self.adaptive_block += 1


@dataclass
class SteerConfig:
    # 조향 발행 주기 (권장 = 20Hz 고정 제어 타이머, CLAUDE.md 10.2 inference_node 구조).
    # 검출은 카메라 콜백(30fps)에서 최신 offset을 래치하고, step() 은 이 주기로 호출한다.
    control_hz: float = 20.0

    # PD 조향 (CLAUDE.md 10.2) — tick 기준 게인. 20Hz 에서 inference_node 값과 동일.
    #  주기를 바꾸면 kp/kd 는 재튜닝 대상(미분이 tick 기준이라 주기 의존).
    steer_kp: float = 0.6
    steer_kd: float = 0.15
    curve_ff: float = 0.30       # 곡률 피드포워드 게인 (곡선 진입을 미리 조향)

    # 슬루/폴백은 "시간 단위"로 정의 → control_hz 로 내부 변환(주기를 바꿔도 거동 유지).
    steer_slew_per_sec: float = 3.0   # 초당 최대 조향 변화. 20Hz 에서 tick당 0.15
    steer_sign: float = 1.0      # 서보 배선 극성. 반대로 돌면 -1.0
    steer_limit: float = 1.0     # 조향 포화

    # 스로틀 / 커브 감속 (B.2)
    cruise_throttle: float = 0.13
    curve_slow: float = 0.5      # 곡률·오프셋에 비례한 감속 강도(0=감속안함,1=최대)
    min_throttle: float = 0.05   # 감속 하한(멈추지 않을 최소 추진)

    # 이탈 방지 폴백 (차선 로스트 시)
    lane_lost_hold_sec: float = 0.5   # 이 시간까지 마지막 조향 유지하며 저속 크리프. 20Hz 에서 10 tick
    lane_lost_throttle: float = 0.05  # 크리프 스로틀
    # hold 시간 초과로도 차선을 못 찾으면 throttle=0 (완주>속도, 무리한 전진 금지)


@dataclass
class LaneResult:
    offset: float               # [-1..1] 음수=차선 중앙 기준 왼쪽, 양수=오른쪽
    curvature: float            # [-1..1] 먼 밴드가 가까운 밴드보다 어느 쪽으로 휘는지
    valid: bool
    valid_bands: int
    band_offsets: List[Optional[float]] = field(default_factory=list)  # near -> far 순


# ---------------------------------------------------------------------------
# 전처리 (노트북 계승)
# ---------------------------------------------------------------------------
def grayscale(frame_bgr: np.ndarray) -> np.ndarray:
    # 실차 카메라는 cv2 디코드(BGR) 기준. 밝기 기반 검출이라 채널 순서는 결과에 영향 없음.
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return gray
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def white_mask(gray: np.ndarray, cfg: LaneConfig) -> np.ndarray:
    """어두운 바닥 위의 밝은 흰 선만 255로 남긴다."""
    if cfg.use_adaptive:
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            cfg.adaptive_block, cfg.adaptive_C,
        )
    else:
        _, mask = cv2.threshold(gray, cfg.fixed_thresh, 255, cv2.THRESH_BINARY)

    if cfg.morph_ksize > 1:
        k = np.ones((cfg.morph_ksize, cfg.morph_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)  # 작은 반사·노이즈 제거
    return mask


# ---------------------------------------------------------------------------
# 곡선 대응 차선 오프셋 (밴드 무게중심)
# ---------------------------------------------------------------------------
def compute_lane(mask: np.ndarray, cfg: LaneConfig) -> LaneResult:
    """
    하단 ROI를 수평 밴드로 나눠, 각 밴드에서 흰 선의 무게중심으로 차선 중앙을 추정한다.
    밴드마다 독립적으로 추정하므로 곡선도 자연스럽게 따라간다.
    band index 0 = 가장 가까운(아래) 밴드, 마지막 = 가장 먼(위) 밴드.
    """
    h, w = mask.shape[:2]
    roi_top = int(h * cfg.roi_top_ratio)
    right = w if cfg.roi_right < 0 else min(cfg.roi_right, w)
    left = max(0, cfg.roi_left)
    roi = mask[roi_top:h, left:right]
    rh, rw = roi.shape[:2]

    # 오프셋 정규화는 "원본 이미지 중앙" 기준 (비대칭 ROI 에서도 offset=0 이 카메라 중심선)
    img_center = w / 2.0
    half = w / 2.0
    band_h = max(1, rh // cfg.num_bands)

    band_offsets: List[Optional[float]] = []
    valid_bands = 0

    for b in range(cfg.num_bands):
        # b=0 이 화면 최하단(가까움)이 되도록 아래에서 위로 자른다
        y1 = rh - b * band_h
        y0 = 0 if b == cfg.num_bands - 1 else rh - (b + 1) * band_h
        band = roi[y0:y1, :]

        ys, xs = np.nonzero(band)
        xs_global = xs + left  # 원본 좌표계로

        cx: Optional[float] = None
        if cfg.split_lanes:
            left_xs = xs_global[xs_global < img_center]
            right_xs = xs_global[xs_global >= img_center]
            left_ok = left_xs.size >= cfg.side_min_px
            right_ok = right_xs.size >= cfg.side_min_px
            if left_ok and right_ok:
                cx = (left_xs.mean() + right_xs.mean()) / 2.0
            elif left_ok:
                # 왼쪽 선만 보임 -> 차선 중앙은 그 오른쪽
                cx = left_xs.mean() + cfg.lane_half_norm * half
            elif right_ok:
                cx = right_xs.mean() - cfg.lane_half_norm * half
        else:
            if xs_global.size >= cfg.valid_min_px:
                cx = float(xs_global.mean())

        if cx is None:
            band_offsets.append(None)
            continue

        off = float(np.clip((cx - img_center) / half, -1.0, 1.0))
        band_offsets.append(off)
        valid_bands += 1

    if valid_bands == 0:
        return LaneResult(0.0, 0.0, False, 0, band_offsets)

    # 가까운 밴드(index 작음)에 더 큰 가중치를 준 가중 평균
    num = 0.0
    den = 0.0
    for i, off in enumerate(band_offsets):
        if off is None:
            continue
        wgt = cfg.num_bands - i  # i=0(가까움) 가장 큼
        num += wgt * off
        den += wgt
    offset = num / den if den > 0 else 0.0

    # 곡률: 먼 밴드가 가까운 밴드보다 어느 쪽으로 휘는지 (앞으로 굽는 방향)
    near = next((o for o in band_offsets if o is not None), None)
    far = next((o for o in reversed(band_offsets) if o is not None), None)
    if near is not None and far is not None:
        curvature = float(np.clip((far - near) / 2.0, -1.0, 1.0))
    else:
        curvature = 0.0

    return LaneResult(offset, curvature, True, valid_bands, band_offsets)


# ---------------------------------------------------------------------------
# 조향 제어기 (PD + 곡률 피드포워드 + 슬루 + 커브 감속 + 이탈 폴백)
# ---------------------------------------------------------------------------
class SteeringController:
    def __init__(self, scfg: SteerConfig):
        self.s = scfg
        self.prev_offset = 0.0
        self.prev_steer = 0.0
        self.lost_time = 0.0  # 차선을 놓친 누적 시간(초)

    def step(self, lane: LaneResult, dt: Optional[float] = None) -> Tuple[float, float, str]:
        """dt: 직전 호출 이후 경과시간(초). None 이면 1/control_hz(권장 20Hz) 사용.
        ROS 20Hz 타이머면 dt 를 생략(고정 주기)하거나 실제 측정 dt 를 넘겨도 된다."""
        s = self.s
        if dt is None or dt <= 0.0:
            dt = 1.0 / s.control_hz

        # --- 차선 로스트: 이탈 방지 폴백 ---
        if not lane.valid:
            self.lost_time += dt
            steer = self.prev_steer  # 마지막 조향 유지 (급격히 직진으로 풀면 코너에서 튀어나감)
            if self.lost_time <= s.lane_lost_hold_sec:
                throttle = s.lane_lost_throttle  # 저속 크리프하며 차선 재포착 대기
            else:
                throttle = 0.0  # 오래 못 찾으면 정지 (완주>속도, 무리한 전진 금지)
            return steer, throttle, "LANE_LOST"

        self.lost_time = 0.0
        offset = lane.offset

        # --- PD + 곡률 피드포워드 (미분은 tick 기준, 고정 주기 가정) ---
        d_offset = offset - self.prev_offset
        raw = (s.steer_kp * offset
               + s.steer_kd * d_offset
               + s.curve_ff * lane.curvature)   # 곡선 진입을 미리 반영(앞바퀴를 먼저 꺾음)
        raw *= s.steer_sign
        raw = float(np.clip(raw, -s.steer_limit, s.steer_limit))

        # --- 슬루 제한(시간당 조향 변화 상한 → dt 로 환산) ---
        max_delta = s.steer_slew_per_sec * dt
        delta = float(np.clip(raw - self.prev_steer, -max_delta, max_delta))
        steer = self.prev_steer + delta

        # --- 커브 감속(B.2): 많이 휘거나 중앙에서 벗어날수록 감속 ---
        curve_mag = min(1.0, abs(lane.curvature) + 0.5 * abs(offset))
        throttle = max(s.min_throttle, s.cruise_throttle * (1.0 - s.curve_slow * curve_mag))

        self.prev_offset = offset
        self.prev_steer = steer
        return steer, throttle, "DRIVE"


# ---------------------------------------------------------------------------
# 프레임 파이프라인 + 시각화
# ---------------------------------------------------------------------------
def process_frame(frame_bgr: np.ndarray, lane_cfg: LaneConfig,
                  controller: SteeringController) -> Tuple[float, float, LaneResult, np.ndarray]:
    gray = grayscale(frame_bgr)
    gray = gaussian_blur(gray, lane_cfg.blur_ksize)
    mask = white_mask(gray, lane_cfg)
    lane = compute_lane(mask, lane_cfg)
    steer, throttle, state = controller.step(lane)
    lane.state = state  # type: ignore[attr-defined]  (디버그용)
    return steer, throttle, lane, mask


def draw_overlay(frame_bgr: np.ndarray, lane: LaneResult, steer: float,
                 throttle: float, lane_cfg: LaneConfig) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cx_img = w // 2

    # 이미지 중심선(파랑)
    cv2.line(out, (cx_img, int(h * lane_cfg.roi_top_ratio)), (cx_img, h), (255, 128, 0), 1)

    # 추정 차선 중앙(초록)
    if lane.valid:
        lane_cx = int(cx_img + lane.offset * (w / 2.0))
        cv2.line(out, (lane_cx, int(h * lane_cfg.roi_top_ratio)), (lane_cx, h), (0, 255, 0), 2)

    # 조향 바(빨강) : 화면 하단 중앙에서 좌/우로
    bar_y = h - 20
    cv2.line(out, (cx_img, bar_y), (int(cx_img + steer * (w / 3.0)), bar_y), (0, 0, 255), 6)

    state = getattr(lane, "state", "")
    txt = f"off={lane.offset:+.2f} cur={lane.curvature:+.2f} steer={steer:+.2f} thr={throttle:.2f} [{state}]"
    cv2.putText(out, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# 테스트 하니스 (이미지 1장 또는 동영상). 실차 없이 오프라인 확인용.
#   python white_lane_follower.py <image_or_video_path> [output_path]
# ---------------------------------------------------------------------------
def _run_image(path: str, out_path: str, lane_cfg: LaneConfig, ctrl: SteeringController):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(path)
    steer, throttle, lane, mask = process_frame(frame, lane_cfg, ctrl)
    overlay = draw_overlay(frame, lane, steer, throttle, lane_cfg)
    cv2.imwrite(out_path, overlay)
    print(f"[image] {getattr(lane, 'state', '')} "
          f"offset={lane.offset:+.3f} curvature={lane.curvature:+.3f} "
          f"steer={steer:+.3f} throttle={throttle:.3f} valid_bands={lane.valid_bands}")
    print(f"        overlay saved -> {out_path}")


def _run_video(path: str, out_path: str, lane_cfg: LaneConfig, ctrl: SteeringController):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        steer, throttle, lane, _ = process_frame(frame, lane_cfg, ctrl)
        writer.write(draw_overlay(frame, lane, steer, throttle, lane_cfg))
        n += 1
    cap.release()
    writer.release()
    print(f"[video] {n} frames processed -> {out_path}")


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "solidWhiteCurve.jpg"
    dst = sys.argv[2] if len(sys.argv) > 2 else "lane_overlay.jpg"

    lane_cfg = LaneConfig()      # 기본 = white_track 프리셋(밝기/split)
    steer_cfg = SteerConfig()
    ctrl = SteeringController(steer_cfg)

    if src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        _run_image(src, dst, lane_cfg, ctrl)
    else:
        if dst == "lane_overlay.jpg":
            dst = "lane_overlay.mp4"
        _run_video(src, dst, lane_cfg, ctrl)


# ===========================================================================
# PROJECT INTEGRATION (실차 D-Racer-Kit 통합 메모)
# ===========================================================================
# CLAUDE.md 구조상 검출과 조향은 노드가 나뉜다. 위 코드도 그 경계로 나눠 넣으면 된다.
#
#   [opencv_node]    카메라 콜백(30fps)에서 compute_lane() 실행 → 최신 offset 래치,
#                    std_msgs/Float32MultiArray 로 발행 -> /lane/offset = [offset, valid, curvature]
#   [inference_node] 20Hz 제어 타이머에서 최신 래치값으로 SteeringController.step() 호출
#                    -> control_msgs/Control 발행 (/control)
#
# ★ 권장 주기: 검출 30fps(카메라 콜백) / 조향 20Hz(고정 타이머).
#   - 조향을 카메라 콜백에 직접 물지 말 것(프레임 지터가 PD·슬루에 실려 곡선에서 조향이 떨림).
#   - 20Hz 타이머는 검출이 한두 프레임 비어도 마지막 래치값으로 계속 돌아 이탈 방지에 유리.
#   - SteerConfig.control_hz(20.0) 가 슬루/로스트 시간 환산 기준. 주기를 바꾸면 kp/kd 재튜닝.
#
# ★ 통합 전 반드시 확인 (알려진 이슈):
#   - 8.2: /lane/offset 은 "절대표기"로 통일. opencv_node 발행 토픽명과 inference_node
#          구독 토픽명이 양쪽 모두 정확히 '/lane/offset' 인지 확인.
#          보드에서:  ros2 topic info /lane/offset   (pub 1 / sub 1 연결 확인)
#          한쪽이 상대표기(lane/offset)면 노드는 정상 기동해도 조향이 계속 중립 유지된다.
#   - 8.1: 커브 감속(B.2)은 기존 inference_node 에서 비활성 상태였다. 위 SteeringController 는
#          curve_slow 로 이를 재활성했다. 안정화 순서 = ① 커브 감속 검증 -> ② 순항속도 신뢰.
#   - 조명: white_track brightness 경로는 광택 바닥 반사를 선으로 오검출할 수 있다(10번).
#          adaptive_block / adaptive_C / roi_top_ratio 를 /opencv/image/edge 보며 실측 튜닝.
#
# ※ 이 파일은 신호등(B.1/B.6)·좌우 표지판(B.3)·ArUco(B.4)·회전 교차로(B.5) 는 다루지 않는다.
#    그건 YOLO26n(inference) 와 cv2.aruco 별도 경로 몫. 여기서는 "흰 차선 추종 + 이탈 방지"만.