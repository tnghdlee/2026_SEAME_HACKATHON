"""inference_node — YOLO26n(ONNX) 추론 + 주행 정책 → /control 발행.

데이터 흐름 (CLAUDE.md 5 / E.5):
    /camera/image/compressed ─► [YOLO26n best.onnx] ─► 시간적 필터 ─┐
    /lane/offset ────────────────────────────────► DrivingPolicy ─┴─► /control
                                                                    └─► /inference/detections(JSON, 디버그)

- 검출은 카메라 콜백(YOLO rate ≈3Hz)에서 policy.on_detections 로 래치 갱신.
- 조향/스로틀은 control timer(control_hz, 기본 20Hz)에서 policy.step(lane) 으로 발행.
- 모델/런타임 부재 시 degraded 모드: 검출 없이 정지(throttle 0)·중립 조향 유지.

토픽·STEER_TRIM 은 vehicle_config.yaml 에서 로드(절대표기 통일, CLAUDE.md 8.2/10).
차선 토픽 기본값은 절대표기 /lane/offset (발행자 opencv_node 와 일치).
"""
import json
import os
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, String
from control_msgs.msg import Control

from inference.driving_policy import DrivingPolicy, LaneSignal, default_params
from inference.yolo_onnx import YoloOnnx


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
        self.declare_parameter('cruise_throttle', dp['cruise_throttle'])
        self.declare_parameter('corner_throttle', dp['corner_throttle'])
        self.declare_parameter('turn_throttle', dp['turn_throttle'])
        self.declare_parameter('lane_lost_throttle', dp['lane_lost_throttle'])
        self.declare_parameter('corner_curvature_threshold', dp['corner_curvature_threshold'])
        self.declare_parameter('curve_hold_decay', dp['curve_hold_decay'])
        self.declare_parameter('steer_sign', dp['steer_sign'])
        self.declare_parameter('steer_kp', dp['steer_kp'])
        self.declare_parameter('steer_kd', dp['steer_kd'])
        self.declare_parameter('steer_slew', dp['steer_slew'])
        self.declare_parameter('turn_bias', dp['turn_bias'])
        self.declare_parameter('commit_lane_weight', dp['commit_lane_weight'])
        self.declare_parameter('commit_steer_slew', dp['commit_steer_slew'])
        self.declare_parameter('fork_commit_frames', dp['fork_commit_frames'])
        self.declare_parameter('sign_margin', dp['sign_margin'])
        self.declare_parameter('sign_conf', dp['sign_conf'])
        self.declare_parameter('drive_direction', dp['drive_direction'])

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value))
        model_path = os.path.expanduser(str(self.get_parameter('model_path').value))
        imgsz = int(self.get_parameter('imgsz').value)
        conf = float(self.get_parameter('conf_threshold').value)
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
            'cruise_throttle': float(self.get_parameter('cruise_throttle').value),
            'corner_throttle': float(self.get_parameter('corner_throttle').value),
            'turn_throttle': float(self.get_parameter('turn_throttle').value),
            'lane_lost_throttle': float(self.get_parameter('lane_lost_throttle').value),
            'corner_curvature_threshold': float(
                self.get_parameter('corner_curvature_threshold').value),
            'curve_hold_decay': float(self.get_parameter('curve_hold_decay').value),
            'steer_trim': steer_trim,
            'steer_sign': float(self.get_parameter('steer_sign').value),
            'steer_kp': float(self.get_parameter('steer_kp').value),
            'steer_kd': float(self.get_parameter('steer_kd').value),
            'steer_slew': float(self.get_parameter('steer_slew').value),
            'turn_bias': float(self.get_parameter('turn_bias').value),
            'commit_lane_weight': float(self.get_parameter('commit_lane_weight').value),
            'commit_steer_slew': float(self.get_parameter('commit_steer_slew').value),
            'fork_commit_frames': int(self.get_parameter('fork_commit_frames').value),
            'sign_margin': float(self.get_parameter('sign_margin').value),
            'sign_conf': float(self.get_parameter('sign_conf').value),
            'drive_direction': float(self.get_parameter('drive_direction').value),
        })
        self.policy = DrivingPolicy(params)
        self.lane = LaneSignal(offset=0.0, valid=False, curvature=0.0)

        # 모델 로드 — 실패 시 degraded 모드(정지).
        self.model = None
        try:
            self.model = YoloOnnx(model_path, imgsz=imgsz, conf_threshold=conf)
            self.get_logger().info(f'YOLO ONNX loaded: {model_path}')
        except Exception as exc:  # noqa: BLE001 - 어떤 실패든 정지로 degrade
            self.get_logger().error(
                f'YOLO 모델 로드 실패 → degraded(정지) 모드: {exc}')

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(CompressedImage, image_topic,
                                 self.image_callback, image_qos)
        self.create_subscription(Float32MultiArray, lane_topic,
                                 self.lane_callback, 10)

        self.control_pub = self.create_publisher(Control, control_topic, 10)
        self.det_pub = None
        if self.publish_detections:
            self.det_pub = self.create_publisher(String, self.detections_topic, 10)

        period = 1.0 / control_hz if control_hz > 0 else 0.05
        self.timer = self.create_timer(period, self.control_tick)

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

    # ---- YOLO rate: 검출 → 정책 래치 ----
    def image_callback(self, msg: CompressedImage):
        if self.model is None:
            return
        import cv2
        import numpy as np

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('압축 이미지 디코드 실패')
            return

        try:
            dets = self.model.infer(frame)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'추론 실패(프레임 스킵): {exc}')
            return

        self.policy.on_detections(dets)

        if self.det_pub is not None:
            payload = [
                {'cls': d.class_id, 'score': round(d.score, 3),
                 'box': [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)]}
                for d in dets
            ]
            msg_out = String()
            msg_out.data = json.dumps(
                {'detections': payload, 'state': self.policy.state_summary()})
            self.det_pub.publish(msg_out)

    # ---- 차선 신호 저장 ----
    def lane_callback(self, msg: Float32MultiArray):
        data = list(msg.data)
        if len(data) >= 3:
            self.lane = LaneSignal(offset=float(data[0]),
                                   valid=bool(data[1] >= 0.5),
                                   curvature=float(data[2]))

    # ---- 제어 rate: 정책 → /control ----
    def control_tick(self):
        steer, throttle = self.policy.step(self.lane)
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
            st = self.policy.state_summary()
            self.get_logger().info(
                f'ctrl: steer={steer:+.3f} throttle={throttle:.3f} '
                f'lane(valid={self.lane.valid} off={self.lane.offset:+.3f} '
                f'curv={self.lane.curvature:+.3f}) corner_hold={st["corner_hold"]:.3f} '
                f'turn_intent={st["turn_intent"]} fork_remaining={st["fork_remaining"]} '
                f'green={st["green_started"]} red={st["red_stopped"]}')


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
