"""inference_node — YOLO26n(ONNX) 추론 + 주행 정책 → /control 발행.

데이터 흐름 (CLAUDE.md 5 / E.5):
    /camera/image/compressed ─► [YOLO26n best.onnx] ─► 시간적 필터 ─┐
    /lane/offset ────────────────────────────────► DrivingPolicy ─┴─► /control
                                                                    └─► /inference/detections(JSON, 디버그)

- 검출은 카메라 콜백(YOLO rate ≈1~3Hz)에서 policy.on_detections 로 래치 갱신.
- 조향/스로틀은 control timer(control_hz, 기본 20Hz)에서 policy.step(lane) 으로 발행.
- 모델/런타임 부재 시 degraded 모드: 검출 없이 정지(throttle 0)·중립 조향 유지.

⚠️ 동시성(중요): YOLO ONNX 추론은 보드 CPU 에서 프레임당 수백 ms 로 느리다. 이를
   control timer 와 같은 단일 스레드(rclpy.spin)에서 돌리면 추론이 타이머를 굶겨
   /control 이 실측 ~1Hz 로 떨어진다(bagfile 확인). 조향 명령이 초당 1회면 차선을
   따라갈 수 없다. 그래서 MultiThreadedExecutor + 콜백 그룹으로 분리한다:
     - image_callback(느린 추론)  : 전용 그룹(중복 추론 방지, MutuallyExclusive).
     - control_tick(20Hz 조향)     : 별도 전용 그룹 → 추론과 무관하게 정시 실행.
     - lane_callback              : 별도 그룹.
   공유 상태(policy, self.lane)는 self._lock 으로 보호하되, 느린 추론은 락 밖에서
   실행해 타이머를 막지 않는다(락 구간은 on_detections/step 등 수 μs 로직뿐).

토픽·STEER_TRIM 은 vehicle_config.yaml 에서 로드(절대표기 통일, CLAUDE.md 8.2/10).
차선 토픽 기본값은 절대표기 /lane/offset (발행자 opencv_node 와 일치).
"""
import json
import os
import threading
from pathlib import Path

import rclpy
import yaml
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, String
from control_msgs.msg import Control

from inference.driving_policy import (DrivingPolicy, LaneSignal, GREENLIGHT,
                                      default_params)
from inference.yolo_onnx import Detection, YoloOnnx
# 순수 상태머신(cv2 비의존) — 안전하게 top-level import. 검출측(aruco_detect)은
# cv2.aruco 의존이라 __init__ 에서 지연 import 하고 실패 시 degraded(ArUco 비활성).
from inference.aruco_stop_policy import ArucoStopPolicy, StopPolicyConfig


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer-Kit/src/config/vehicle_config.yaml'


def get_default_model_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'models' / 'best.onnx'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer-Kit/models/best.onnx'


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # --- 인프라 param ---
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('model_path', get_default_model_path())
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('control_hz', 20.0)
        self.declare_parameter('lane_offset_topic', '/lane/offset')
        self.declare_parameter('detections_topic', '/inference/detections')
        self.declare_parameter('publish_detections', True)

        # --- 주행 정책 param (기본값은 default_params, launch 가 덮어씀) ---
        dp = default_params()
        self.declare_parameter('start_confirm_frames', dp['start_confirm_frames'])
        self.declare_parameter('confirm_frames', dp['confirm_frames'])
        self.declare_parameter('stop_confirm_frames', dp['stop_confirm_frames'])
        self.declare_parameter('require_green_start', dp['require_green_start'])
        self.declare_parameter('green_resumes_from_red', dp['green_resumes_from_red'])
        self.declare_parameter('redlight_max_y_ratio', dp['redlight_max_y_ratio'])
        self.declare_parameter('redlight_max_h_ratio', dp['redlight_max_h_ratio'])
        self.declare_parameter('green_start_conf', dp['green_start_conf'])
        # --- 출발 초록불 HSV 색 폴백 (먼 신호등 대비, 출발 전에만 활성) ---
        self.declare_parameter('green_hsv_fallback', True)
        # 초록 HSV 범위(OpenCV H 0~180). 신호등 초록은 밝고 채도 높음.
        self.declare_parameter('green_hsv_lower', [40, 80, 80])
        self.declare_parameter('green_hsv_upper', [90, 255, 255])
        # 초록 blob 을 찾을 세로 ROI(프레임 높이 비율). 신호등은 상단에 있으므로
        # 하단(바닥/잔디)을 배제해 오검출을 줄인다.
        self.declare_parameter('green_roi_top_ratio', 0.0)
        self.declare_parameter('green_roi_bottom_ratio', 0.6)
        # blob 최소 면적(px²). 먼 신호등은 작으므로 작게. 배경 초록 오검출 시 키움.
        self.declare_parameter('green_min_area', 8.0)
        # 폴백 blob 을 GREENLIGHT 로 합성할 때 부여하는 점수(green_start_conf 초과).
        self.declare_parameter('green_blob_score', 0.5)
        self.declare_parameter('cruise_throttle', dp['cruise_throttle'])
        self.declare_parameter('corner_throttle', dp['corner_throttle'])
        self.declare_parameter('turn_throttle', dp['turn_throttle'])
        self.declare_parameter('lane_lost_throttle', dp['lane_lost_throttle'])
        self.declare_parameter('steer_throttle', dp['steer_throttle'])
        self.declare_parameter('steer_throttle_threshold', dp['steer_throttle_threshold'])
        self.declare_parameter('corner_curvature_threshold', dp['corner_curvature_threshold'])
        self.declare_parameter('curve_hold_decay', dp['curve_hold_decay'])
        self.declare_parameter('steer_sign', dp['steer_sign'])
        self.declare_parameter('steer_kp', dp['steer_kp'])
        self.declare_parameter('steer_kd', dp['steer_kd'])
        self.declare_parameter('curve_ff', dp['curve_ff'])
        self.declare_parameter('steer_slew', dp['steer_slew'])
        self.declare_parameter('turn_bias', dp['turn_bias'])
        self.declare_parameter('commit_lane_weight', dp['commit_lane_weight'])
        self.declare_parameter('commit_steer_slew', dp['commit_steer_slew'])
        self.declare_parameter('fork_commit_frames', dp['fork_commit_frames'])
        self.declare_parameter('sign_margin', dp['sign_margin'])
        self.declare_parameter('sign_conf', dp['sign_conf'])
        self.declare_parameter('start_straight_frames', dp['start_straight_frames'])
        self.declare_parameter('drive_direction', dp['drive_direction'])
        # 출발 킥스타트: 초록불 확정 직후 조향 없이 고정 throttle 로 직진하는 구간.
        # 지속시간은 '초'로 받아 control_hz 로 프레임 환산(제어율에 무관하게 일정).
        self.declare_parameter('start_kick_throttle', dp['start_kick_throttle'])
        self.declare_parameter('start_kick_seconds', 2.0)

        # --- ArUco 동적 장애물 정지/재출발 param (B.4, aruco_detect/aruco_stop_policy) ---
        # aruco_enabled=False 면 검출·정지 로직 전체 비활성(ArUco 무시).
        self.declare_parameter('aruco_enabled', True)
        self.declare_parameter('aruco_dict', 'DICT_6X6_50')
        # 규정 마커 ID 목록. 빈 배열([])이면 ID 무관(아무 마커나 인정).
        self.declare_parameter('aruco_target_ids', [3])
        # 근접 게이팅: 마커 면적/프레임 면적이 이 값 이상일 때만 정지(멀면 무시).
        self.declare_parameter('aruco_min_area_ratio', 0.0)
        # ROI 게이팅: 마커 중심이 이 정규화 사각형[x0,y0,x1,y1] 안일 때만 인정.
        # 빈 배열([])이면 전체 화면. 기본 = 하단 60%·가로 중앙 60%(주행 경로).
        self.declare_parameter('aruco_roi_norm', [0.2, 0.4, 0.8, 1.0])
        # 정지 진입(민감·작게) / 재출발 확인(보수·크게) — 비대칭 히스테리시스(9.5).
        # 카메라 프레임 rate 단위(image_callback 마다 1프레임).
        self.declare_parameter('aruco_stop_confirm_frames', 2)
        self.declare_parameter('aruco_clear_confirm_frames', 6)

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value))
        model_path = os.path.expanduser(str(self.get_parameter('model_path').value))
        imgsz = int(self.get_parameter('imgsz').value)
        conf = float(self.get_parameter('conf_threshold').value)
        green_start_conf = float(self.get_parameter('green_start_conf').value)
        # HSV 초록 폴백 설정을 인스턴스에 보관(image_callback 에서 사용).
        self.green_hsv_fallback = bool(self.get_parameter('green_hsv_fallback').value)
        self._green_hsv_lower = [int(v) for v in
                                 self.get_parameter('green_hsv_lower').value]
        self._green_hsv_upper = [int(v) for v in
                                 self.get_parameter('green_hsv_upper').value]
        self._green_roi_top = float(self.get_parameter('green_roi_top_ratio').value)
        self._green_roi_bottom = float(self.get_parameter('green_roi_bottom_ratio').value)
        self._green_min_area = float(self.get_parameter('green_min_area').value)
        self._green_blob_score = float(self.get_parameter('green_blob_score').value)
        control_hz = float(self.get_parameter('control_hz').value)
        self.control_hz = control_hz if control_hz > 0 else 20.0
        lane_topic = str(self.get_parameter('lane_offset_topic').value)
        self.detections_topic = str(self.get_parameter('detections_topic').value)
        self.publish_detections = bool(self.get_parameter('publish_detections').value)

        # vehicle_config: 토픽·STEER_TRIM.
        config = self._load_config()
        image_topic = str(config.get('IMAGE_TOPIC', '/camera/image/compressed'))
        control_topic = str(config.get('CONTROL_TOPIC', '/control'))
        steer_trim = float(config.get('STEER_TRIM', 0.0))

        # 정책 파라미터 취합.
        params = default_params()
        params.update({
            'conf_threshold': conf,
            'start_confirm_frames': int(self.get_parameter('start_confirm_frames').value),
            'confirm_frames': int(self.get_parameter('confirm_frames').value),
            'stop_confirm_frames': int(self.get_parameter('stop_confirm_frames').value),
            'require_green_start': bool(self.get_parameter('require_green_start').value),
            'green_resumes_from_red': bool(
                self.get_parameter('green_resumes_from_red').value),
            'redlight_max_y_ratio': float(
                self.get_parameter('redlight_max_y_ratio').value),
            'redlight_max_h_ratio': float(
                self.get_parameter('redlight_max_h_ratio').value),
            'green_start_conf': green_start_conf,
            'cruise_throttle': float(self.get_parameter('cruise_throttle').value),
            'corner_throttle': float(self.get_parameter('corner_throttle').value),
            'turn_throttle': float(self.get_parameter('turn_throttle').value),
            'lane_lost_throttle': float(self.get_parameter('lane_lost_throttle').value),
            'steer_throttle': float(self.get_parameter('steer_throttle').value),
            'steer_throttle_threshold': float(
                self.get_parameter('steer_throttle_threshold').value),
            'corner_curvature_threshold': float(
                self.get_parameter('corner_curvature_threshold').value),
            'curve_hold_decay': float(self.get_parameter('curve_hold_decay').value),
            'steer_trim': steer_trim,
            'steer_sign': float(self.get_parameter('steer_sign').value),
            'steer_kp': float(self.get_parameter('steer_kp').value),
            'steer_kd': float(self.get_parameter('steer_kd').value),
            'curve_ff': float(self.get_parameter('curve_ff').value),
            'steer_slew': float(self.get_parameter('steer_slew').value),
            'turn_bias': float(self.get_parameter('turn_bias').value),
            'commit_lane_weight': float(self.get_parameter('commit_lane_weight').value),
            'commit_steer_slew': float(self.get_parameter('commit_steer_slew').value),
            'fork_commit_frames': int(self.get_parameter('fork_commit_frames').value),
            'sign_margin': float(self.get_parameter('sign_margin').value),
            'sign_conf': float(self.get_parameter('sign_conf').value),
            'start_straight_frames': int(
                self.get_parameter('start_straight_frames').value),
            'drive_direction': float(self.get_parameter('drive_direction').value),
            'start_kick_throttle': float(
                self.get_parameter('start_kick_throttle').value),
            # 초 → 제어 프레임 환산(2s × control_hz). 최소 0.
            'start_kick_frames': max(0, int(round(
                float(self.get_parameter('start_kick_seconds').value)
                * self.control_hz))),
        })
        self.policy = DrivingPolicy(params)
        self.lane = LaneSignal(offset=0.0, valid=False, curvature=0.0)
        # 공유 상태(policy/self.lane) 보호. 느린 추론은 이 락 밖에서 실행한다.
        self._lock = threading.Lock()

        # 모델 로드 — 실패 시 degraded 모드(정지).
        # YOLO 출력 임계값은 클래스별 최소 임계값(=초록불 완화값)까지 낮춰,
        # 낮은 신뢰도의 먼 초록불 검출도 정책까지 전달되게 한다. redlight/표지판은
        # on_detections 에서 conf_threshold 로 다시 필터되므로 안전하다.
        yolo_conf = min(conf, green_start_conf)
        self.model = None
        try:
            self.model = YoloOnnx(model_path, imgsz=imgsz, conf_threshold=yolo_conf)
            self.get_logger().info(
                f'YOLO ONNX loaded: {model_path} (yolo_conf={yolo_conf:.3f}, '
                f'policy_conf={conf:.3f}, green_start_conf={green_start_conf:.3f})')
        except Exception as exc:  # noqa: BLE001 - 어떤 실패든 정지로 degrade
            self.get_logger().error(
                f'YOLO 모델 로드 실패 → degraded(정지) 모드: {exc}')

        # --- ArUco 동적 장애물(B.4) 검출기 + 정지 상태머신 ---
        # 검출측(aruco_detect)은 cv2.aruco 의존이라 지연 import + try/except 로
        # degrade(실패 시 self.aruco_detector=None → 장애물 무시, 정지 로직 비활성).
        # 상태머신은 순수 로직이라 항상 생성. self._aruco_blocked 는 control_tick 이
        # throttle 을 0.0 으로 덮어쓸지 결정하는 공유 플래그(락 보호).
        self._aruco_blocked = False
        self.aruco_detector = None
        self.aruco_cfg = None
        self.aruco_stop = None
        self._aruco_mod = None
        aruco_enabled = bool(self.get_parameter('aruco_enabled').value)
        if aruco_enabled:
            try:
                from inference import aruco_detect as aruco_mod
                target_ids = tuple(int(v) for v in
                                   self.get_parameter('aruco_target_ids').value)
                roi = [float(v) for v in
                       self.get_parameter('aruco_roi_norm').value]
                self.aruco_cfg = aruco_mod.ArucoConfig(
                    dict_name=str(self.get_parameter('aruco_dict').value),
                    # 빈 배열 → None(ID/ROI 게이팅 미적용).
                    target_ids=target_ids if target_ids else None,
                    min_area_ratio=float(
                        self.get_parameter('aruco_min_area_ratio').value),
                    roi_norm=tuple(roi) if len(roi) == 4 else None,
                )
                self.aruco_detector = aruco_mod.create_detector(self.aruco_cfg)
                self._aruco_mod = aruco_mod
                self.aruco_stop = ArucoStopPolicy(StopPolicyConfig(
                    stop_confirm_frames=int(
                        self.get_parameter('aruco_stop_confirm_frames').value),
                    clear_confirm_frames=int(
                        self.get_parameter('aruco_clear_confirm_frames').value),
                ))
                self.get_logger().info(
                    f'ArUco 장애물 검출 활성: dict={self.aruco_cfg.dict_name}, '
                    f'ids={self.aruco_cfg.target_ids}, roi={self.aruco_cfg.roi_norm}, '
                    f'min_area_ratio={self.aruco_cfg.min_area_ratio}')
            except Exception as exc:  # noqa: BLE001 - 검출기 생성 실패 시 ArUco 비활성
                self.aruco_detector = None
                self.aruco_stop = None
                self.get_logger().error(
                    f'ArUco 검출기 생성 실패 → 장애물 정지 비활성: {exc}')
        else:
            self.get_logger().info('ArUco 장애물 검출 비활성(aruco_enabled=False)')

        # 센서 영상용 QoS: 최신 프레임만(depth=1) + BEST_EFFORT.
        # 카메라/opencv 발행자와 일치. 추론이 느려도 낡은 프레임이 큐잉되지 않는다.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # 콜백 그룹 분리 → MultiThreadedExecutor 에서 각기 다른 스레드로 동시 실행.
        # 느린 추론(image)이 20Hz 조향 타이머(control)를 굶기지 않게 한다.
        self._img_group = MutuallyExclusiveCallbackGroup()   # 추론 중복 방지
        self._ctrl_group = MutuallyExclusiveCallbackGroup()  # 정시 조향
        self._lane_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(CompressedImage, image_topic,
                                 self.image_callback, image_qos,
                                 callback_group=self._img_group)
        self.create_subscription(Float32MultiArray, lane_topic,
                                 self.lane_callback, 10,
                                 callback_group=self._lane_group)

        self.control_pub = self.create_publisher(Control, control_topic, 10)
        self.det_pub = None
        if self.publish_detections:
            self.det_pub = self.create_publisher(String, self.detections_topic, 10)

        period = 1.0 / control_hz if control_hz > 0 else 0.05
        self.timer = self.create_timer(period, self.control_tick,
                                       callback_group=self._ctrl_group)

        self.get_logger().info(
            f'inference_node started: image_topic={image_topic}, '
            f'lane_topic={lane_topic}, control_topic={control_topic}, '
            f'control_hz={control_hz}, model={"loaded" if self.model else "DEGRADED"}, '
            f'require_green_start={params["require_green_start"]}, '
            f'steer_trim={steer_trim}')

    # ---- config ----
    def _load_config(self):
        if not os.path.exists(self.vehicle_config_file):
            self.get_logger().warning(
                f'vehicle_config 없음: {self.vehicle_config_file} — 기본 토픽 사용')
            return {}
        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as stream:
                return yaml.safe_load(stream) or {}
        except (OSError, yaml.YAMLError) as exc:
            self.get_logger().warning(f'vehicle_config 읽기 실패: {exc}')
            return {}

    # ---- 출발 초록불 HSV 색 폴백 ----
    def _detect_green_blob(self, frame):
        """프레임 상단 ROI 에서 HSV 초록 blob 을 찾아 GREENLIGHT Detection 으로 합성.

        먼 신호등은 YOLO 신뢰도가 낮아 놓칠 수 있으나 초록색은 살아있다. 상단
        ROI 로 제한해 잔디/바닥 등 배경 초록을 배제하고, 최소 면적을 넘는 blob 만
        신호등으로 간주한다. 반환: Detection(GREENLIGHT, ...) 또는 None.
        출발 전에만 호출된다(image_callback 가 green_started 로 게이팅).
        """
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        y0 = max(0, int(self._green_roi_top * h))
        y1 = min(h, int(self._green_roi_bottom * h))
        if y1 <= y0:
            return None
        roi = frame[y0:y1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lo = np.array(self._green_hsv_lower, dtype=np.uint8)
        hi = np.array(self._green_hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < self._green_min_area:
            return None
        x, y, bw, bh = cv2.boundingRect(c)
        # ROI 좌표 → 원본 프레임 좌표(y0 오프셋 보정).
        return Detection(GREENLIGHT, self._green_blob_score,
                         float(x), float(y + y0), float(x + bw), float(y + bh + y0))

    # ---- ArUco 동적 장애물(B.4): 프레임당 검출 → 정지 상태머신 ----
    def _update_aruco(self, frame, cv2):
        """그레이스케일 변환 → obstacle_present(3중 게이팅) → ArucoStopPolicy 갱신.

        결과를 self._aruco_blocked(락 보호)로 저장한다. control_tick 이 이 값을 읽어
        정지 시 throttle 을 0.0 으로 덮어쓴다. 검출/게이팅 파라미터는 aruco_cfg,
        정지/재출발 히스테리시스는 aruco_stop 이 담당(9.5: 정지 민감·재출발 보수).
        """
        if self.aruco_detector is None or self.aruco_stop is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            present, _hits = self._aruco_mod.obstacle_present(
                gray, self.aruco_detector, self.aruco_cfg)
        except Exception as exc:  # noqa: BLE001 - 검출 실패 시 프레임 스킵(상태 유지)
            self.get_logger().warning(f'ArUco 검출 실패(프레임 스킵): {exc}')
            return
        with self._lock:
            allow = self.aruco_stop.update(present)
            self._aruco_blocked = not allow

    # ---- YOLO rate: 검출 → 정책 래치 ----
    def image_callback(self, msg: CompressedImage):
        # 모델·ArUco 모두 없으면 처리할 게 없으므로 조기 반환(degraded).
        if self.model is None and self.aruco_detector is None:
            return
        import cv2
        import numpy as np

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('압축 이미지 디코드 실패')
            return

        # ArUco 동적 장애물(B.4): YOLO 와 독립적으로 매 프레임 검출·상태머신 갱신.
        # 결과(self._aruco_blocked)는 control_tick 이 throttle 을 0.0 으로 덮어쓸지 결정.
        self._update_aruco(frame, cv2)

        if self.model is None:
            return

        # 느린 추론은 락 밖에서 — control 타이머 스레드를 막지 않는다.
        try:
            dets = self.model.infer(frame)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'추론 실패(프레임 스킵): {exc}')
            return

        # 출발 대기 중이면 HSV 색 기반 초록 blob 을 폴백으로 추가한다. 먼 신호등은
        # YOLO 신뢰도가 낮아 놓칠 수 있으나 초록색 자체는 살아있어 색으로 잡힌다.
        # 출발 후(green_started)엔 배경 초록 오검출 방지를 위해 비활성.
        if self.green_hsv_fallback and not self.policy.green_started:
            blob = self._detect_green_blob(frame)
            if blob is not None:
                dets = list(dets) + [blob]

        # 정책 래치 갱신·상태 스냅샷은 락 안에서(수 μs). 프레임 크기를 넘겨
        # 빨간불 바닥 오검출(B.4) 기하 게이팅을 활성화한다.
        frame_h, frame_w = frame.shape[:2]
        with self._lock:
            self.policy.on_detections(dets, frame_height=frame_h, frame_width=frame_w)
            state = self.policy.state_summary()

        if self.det_pub is not None:
            payload = [
                {'cls': d.class_id, 'score': round(d.score, 3),
                 'box': [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)]}
                for d in dets
            ]
            msg_out = String()
            msg_out.data = json.dumps({'detections': payload, 'state': state})
            self.det_pub.publish(msg_out)

    # ---- 차선 신호 저장 ----
    def lane_callback(self, msg: Float32MultiArray):
        data = list(msg.data)
        if len(data) >= 3:
            sig = LaneSignal(offset=float(data[0]),
                             valid=bool(data[1] >= 0.5),
                             curvature=float(data[2]))
            with self._lock:
                self.lane = sig

    # ---- 제어 rate: 정책 → /control ----
    def control_tick(self):
        # 정책 계산·상태 스냅샷은 락 안에서(수 μs). 추론과 병렬로 정시 실행된다.
        with self._lock:
            lane = self.lane
            steer, throttle = self.policy.step(lane)
            st = self.policy.state_summary()
            aruco_blocked = self._aruco_blocked

        # ArUco 동적 장애물(B.4) 하드 정지: 정지 상태면 throttle 을 0.0 으로 덮어쓴다.
        # 조향은 유지(정지 중이라 무해). 빨간불 하드 정지와 동일하게 최우선 정지.
        if aruco_blocked:
            throttle = 0.0

        out = Control()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'inference'
        out.steering = float(steer)
        out.throttle = float(throttle)
        self.control_pub.publish(out)

        # 진단 로깅 — 약 1초마다(제어 20Hz 기준) 조향/스로틀/래치 상태 출력.
        # 갈림길에서 turn_intent 가 래치되고 steer 가 실제로 꺾이는지 확인용.
        self._tick_log = getattr(self, '_tick_log', 0) + 1
        if self._tick_log >= int(self.control_hz):
            self._tick_log = 0
            self.get_logger().info(
                f'ctrl: steer={steer:+.3f} throttle={throttle:.3f} '
                f'lane(valid={lane.valid} off={lane.offset:+.3f} '
                f'curv={lane.curvature:+.3f}) corner_hold={st["corner_hold"]:.3f} '
                f'turn_intent={st["turn_intent"]} fork_remaining={st["fork_remaining"]} '
                f'kick={st["start_kick_remaining"]} '
                f'green={st["green_started"]} red={st["red_stopped"]} '
                f'aruco_stop={aruco_blocked}')


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    # 최소 3스레드: image(추론)·control(20Hz)·lane 이 각기 다른 스레드로 동시 실행.
    # 단일 스레드면 느린 추론이 조향 타이머를 굶겨 /control 이 ~1Hz 로 떨어진다.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
