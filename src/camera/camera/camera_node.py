import os
import re
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
import yaml


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


# flip_method(문자열) -> cv2.rotate 코드 매핑. 'none'/'' 이면 회전 없음.
_FLIP_MAP = {
    'rotate-180': cv2.ROTATE_180,
    '180': cv2.ROTATE_180,
    'clockwise': cv2.ROTATE_90_CLOCKWISE,
    'rotate-90': cv2.ROTATE_90_CLOCKWISE,
    'counterclockwise': cv2.ROTATE_90_COUNTERCLOCKWISE,
    'rotate-270': cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # ROS parameters
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('publish_topic', 'camera/image/compressed')
        self.declare_parameter('publish_hz', 30.0)
        self.declare_parameter('camera_device', '/dev/video0')
        self.declare_parameter('usb_camera_device', '/dev/video1')
        self.declare_parameter('mipi_camera_device', '/dev/video0')
        self.declare_parameter('flip_method', 'none')
        # q90 은 데이터량이 커 인코딩·전송 지연을 늘린다. 비전용은 80 이면 충분.
        self.declare_parameter('jpeg_quality', 80)
        # 매 프레임 로깅은 30Hz I/O 로 콜백을 느리게 만든다. 기본 off.
        self.declare_parameter('debug_log', False)

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        publish_topic = str(self.get_parameter('publish_topic').value)
        publish_hz = float(self.get_parameter('publish_hz').value)
        if publish_hz <= 0.0:
            raise ValueError('publish_hz must be greater than 0')
        default_camera_device = str(self.get_parameter('camera_device').value)
        usb_camera_device = str(self.get_parameter('usb_camera_device').value)
        mipi_camera_device = str(self.get_parameter('mipi_camera_device').value)
        flip_method = str(self.get_parameter('flip_method').value)
        jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        if not 0 <= jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')
        self.debug_log = bool(self.get_parameter('debug_log').value)
        self.publish_hz = publish_hz
        self.jpeg_quality = jpeg_quality

        self.image_width, self.image_height = self.load_image_size()
        self.usb_cam_enabled, self.mipi_cam_enabled = self.load_camera_source_flags()
        usb_camera_device, mipi_camera_device = self.load_camera_device_overrides(
            usb_camera_device,
            mipi_camera_device,
        )
        if self.usb_cam_enabled:
            self.camera_source = 'usb'
            camera_device = usb_camera_device or default_camera_device
        else:
            self.camera_source = 'mipi'
            camera_device = mipi_camera_device or default_camera_device

        self.camera_device = camera_device
        self.flip_method = flip_method
        # 회전 코드 사전 계산. 알 수 없는 값은 회전 없음으로 처리.
        self.rotate_code = _FLIP_MAP.get(str(flip_method).strip().lower(), None)

        # 센서 스트림용 QoS: 최신 프레임만 유지(depth=1) + BEST_EFFORT.
        # RELIABLE+depth10 은 느린 구독자에서 재전송/큐잉으로 지연을 누적시킨다.
        # 지연보다 최신성이 중요한 카메라 영상은 BEST_EFFORT 가 표준.
        self.image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher_ = self.create_publisher(CompressedImage, publish_topic, self.image_qos)
        self.cap = None
        self.pipeline = None
        if not self.open_capture():
            raise RuntimeError(
                'Failed to open camera '
                f'(source={self.camera_source}, device={camera_device}, '
                f'width={self.image_width}, height={self.image_height})'
            )

        self.timer = self.create_timer(1.0 / self.publish_hz, self.timer_callback)
        self.get_logger().info('\n'
            f'[Camera Node] : topic={publish_topic} \n'
            f'[camera source] : {self.camera_source} \n'
            f'[width] : {self.image_width}, [height] : {self.image_height} \n'
            f'[camera_device] : {camera_device} \n'
            f'[flip_method] : {flip_method} \n'
            f'[jpeg_quality] : {self.jpeg_quality} \n'
            f'[vehicle_config_file] : {self.vehicle_config_file} \n'
            f'[debug_log] : {self.debug_log} \n'
        )

    def load_image_size(self):
        default_size = (640, 480)
        if not os.path.exists(self.vehicle_config_file):
            return default_size

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as config_stream:
                config_data = yaml.safe_load(config_stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return default_size

        image_width = int(config_data.get('IMAGE_WIDTH', default_size[0]))
        image_height = int(config_data.get('IMAGE_HEIGHT', default_size[1]))
        return image_width, image_height

    def load_camera_source_flags(self):
        # Backward-compatible default: MIPI enabled.
        default_usb_cam = False
        default_mipi_cam = True

        if not os.path.exists(self.vehicle_config_file):
            return default_usb_cam, default_mipi_cam

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as config_stream:
                config_data = yaml.safe_load(config_stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return default_usb_cam, default_mipi_cam

        usb_cam = bool(config_data.get('USB_CAM', default_usb_cam))
        mipi_cam = bool(config_data.get('MIPI_CAM', default_mipi_cam))

        if usb_cam and mipi_cam:
            raise ValueError('Only one of USB_CAM or MIPI_CAM can be true.')
        if not usb_cam and not mipi_cam:
            raise ValueError('One of USB_CAM or MIPI_CAM must be true.')

        return usb_cam, mipi_cam

    @staticmethod
    def _device_index(device_path):
        """'/dev/video1' -> 1. 숫자를 못 찾으면 원본 문자열을 그대로 반환."""
        match = re.search(r'(\d+)\s*$', str(device_path))
        if match:
            return int(match.group(1))
        return device_path

    def open_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.usb_cam_enabled:
            return self._open_usb_v4l2()
        return self._open_mipi_gstreamer()

    def _open_usb_v4l2(self):
        """USB 웹캠(C920 등)을 OpenCV V4L2 백엔드로 직접 오픈.

        GStreamer 미포함 OpenCV(GStreamer: NO)에서도 동작한다.
        MJPG를 우선 시도(USB 대역폭 절감)하고, 실패 시 기본 포맷으로 폴백한다.
        """
        index = self._device_index(self.camera_device)
        # (라벨, FOURCC) 후보. FOURCC=None 이면 카메라 기본 포맷(YUYV 등) 사용.
        candidates = [('MJPG', 'MJPG'), ('default', None)]

        for label, fourcc in candidates:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                self.get_logger().warning(
                    f'Failed to open /dev/video{index} via V4L2 (fourcc={label})'
                )
                continue

            if fourcc is not None:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            # 드라이버 버퍼를 1프레임으로 줄여 낡은 프레임 누적(지연)을 방지.
            # 타이머 콜백이 캡처 FPS를 못 따라가도 항상 최신 프레임에 가깝게 읽는다.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # 설정값으로 요청. 미지원 해상도면 드라이버가 가장 가까운 값으로 맞춘다.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
            cap.set(cv2.CAP_PROP_FPS, self.publish_hz)

            # 실제로 프레임을 읽을 수 있는지 검증.
            ok, frame = cap.read()
            if ok and frame is not None:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.cap = cap
                self.pipeline = (
                    f'V4L2 /dev/video{index} fourcc={label} '
                    f'capture={actual_w}x{actual_h} -> output={self.image_width}x{self.image_height}'
                )
                self.get_logger().info(f'Camera capture opened: {self.pipeline}')
                if (actual_w, actual_h) != (self.image_width, self.image_height):
                    self.get_logger().info(
                        f'Capture size {actual_w}x{actual_h} differs from configured '
                        f'{self.image_width}x{self.image_height}; frames will be resized.'
                    )
                return True

            cap.release()
            self.get_logger().warning(
                f'Opened /dev/video{index} but could not read a frame (fourcc={label})'
            )

        self.cap = None
        self.pipeline = None
        return False

    def _open_mipi_gstreamer(self):
        """MIPI 카메라 경로 (GStreamer 파이프라인).

        주의: 이 경로는 cv2.CAP_GSTREAMER 를 사용하므로, OpenCV가 GStreamer 지원
        없이 빌드된 경우(GStreamer: NO) 열리지 않는다. 현재 차량은 USB 전용이므로
        이 경로는 사용하지 않는다.
        """
        pipeline = (
            f"v4l2src device={self.camera_device} io-mode=2 ! "
            f"video/x-raw,format=NV12,width={self.image_width},height={self.image_height},framerate=30/1 ! "
            f"videoconvert ! videoflip method={self.flip_method} ! "
            "video/x-raw,format=BGR ! appsink sync=false drop=true max-buffers=1"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self.cap = cap
            self.pipeline = pipeline
            self.get_logger().info(f'Camera capture opened with pipeline: {pipeline}')
            return True

        cap.release()
        self.get_logger().warning(
            'Failed to open MIPI GStreamer pipeline. '
            'If OpenCV was built with "GStreamer: NO", this path cannot work. '
            f'pipeline: {pipeline}'
        )
        self.cap = None
        self.pipeline = None
        return False

    def load_camera_device_overrides(self, default_usb_camera_device, default_mipi_camera_device):
        if not os.path.exists(self.vehicle_config_file):
            return default_usb_camera_device, default_mipi_camera_device

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as config_stream:
                config_data = yaml.safe_load(config_stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return default_usb_camera_device, default_mipi_camera_device

        usb_camera_device = str(
            config_data.get('USB_CAM_DEVICE', default_usb_camera_device)
        ).strip()
        mipi_camera_device = str(
            config_data.get('MIPI_CAM_DEVICE', default_mipi_camera_device)
        ).strip()
        return usb_camera_device, mipi_camera_device

    def timer_callback(self):
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().warning('Camera capture is not opened')
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warning('Failed to read frame')
            return

        # V4L2 직접 오픈에서는 GStreamer videoflip 대신 여기서 회전 적용.
        if self.usb_cam_enabled and self.rotate_code is not None:
            frame = cv2.rotate(frame, self.rotate_code)

        # 설정 해상도와 다르면(미지원 해상도로 드라이버가 대체한 경우) 맞춰준다.
        if frame.shape[1] != self.image_width or frame.shape[0] != self.image_height:
            frame = cv2.resize(
                frame,
                (self.image_width, self.image_height),
                interpolation=cv2.INTER_AREA,
            )

        success, encoded = cv2.imencode(
            '.jpg',
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not success:
            self.get_logger().warning('Failed to encode frame as JPEG')
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()

        self.publisher_.publish(msg)
        if self.debug_log:
            self.get_logger().info(f'Published frame: {len(msg.data)} bytes')

    def destroy_node(self):
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()