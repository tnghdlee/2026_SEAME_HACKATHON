import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

from opencv.lane_detect import compute_lane_offset


class OpenCvNode(Node):
    def __init__(self):
        super().__init__('opencv_node')

        self.declare_parameter('subscribe_topic', '/camera/image/compressed')
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('debug_log', True)

        # --- 차선 추종 오프셋 (CLAUDE.md 8.1) ---
        # inference_node 가 조향에 융합하도록 Float32MultiArray
        # [offset, valid, curvature] 로 발행. ROI 와 이진화를 실트랙에서 튜닝할 수
        # 있도록 ROS param 으로 노출 (CLAUDE.md 9.4).
        self.declare_parameter('publish_lane', True)
        self.declare_parameter('lane_offset_topic', '/lane/offset')
        self.declare_parameter('roi_top', 50)     # vehicle_config 의 ROI_TOP 과 일치
        self.declare_parameter('roi_left', 0)
        self.declare_parameter('lane_num_bands', 3)
        self.declare_parameter('lane_valid_min_px', 40)
        # 'light' = 어두운 바닥 위 밝은 차선, 'dark' = 밝은 바닥 위 어두운 라인.
        # 실트랙 프레임을 확인한 뒤 설정할 것.
        self.declare_parameter('lane_polarity', 'light')
        # 검출 방식: 'color'=HSV 색 마스크(유색 라인에 강건), 'brightness'=명암.
        # 흰 바닥 위 주황 라인 트랙은 'color' + 아래 주황 HSV 범위 권장.
        self.declare_parameter('lane_method', 'color')
        # 주황 라인 기본 HSV 범위(OpenCV H 0~180). 실측: 주황≈H5~22.
        self.declare_parameter('lane_hsv_lower', [5, 80, 80])
        self.declare_parameter('lane_hsv_upper', [22, 255, 255])

        subscribe_topic = str(self.get_parameter('subscribe_topic').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.debug_log = bool(self.get_parameter('debug_log').value)

        self.publish_lane = bool(self.get_parameter('publish_lane').value)
        lane_offset_topic = str(self.get_parameter('lane_offset_topic').value)
        self.roi_top = int(self.get_parameter('roi_top').value)
        self.roi_left = int(self.get_parameter('roi_left').value)
        self.lane_num_bands = int(self.get_parameter('lane_num_bands').value)
        self.lane_valid_min_px = int(self.get_parameter('lane_valid_min_px').value)
        self.lane_polarity = str(self.get_parameter('lane_polarity').value)
        self.lane_method = str(self.get_parameter('lane_method').value)
        self.lane_hsv_lower = [int(v) for v in self.get_parameter('lane_hsv_lower').value]
        self.lane_hsv_upper = [int(v) for v in self.get_parameter('lane_hsv_upper').value]

        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            subscribe_topic,
            self.image_callback,
            image_qos,
        )

        self.gray_pub = self.create_publisher(
            CompressedImage,
            '/opencv/image/grayscale',
            image_qos,
        )
        self.blur_pub = self.create_publisher(
            CompressedImage,
            '/opencv/image/blur',
            image_qos,
        )
        self.edge_pub = self.create_publisher(
            CompressedImage,
            '/opencv/image/edge',
            image_qos,
        )

        self.lane_pub = None
        if self.publish_lane:
            self.lane_pub = self.create_publisher(
                Float32MultiArray,
                lane_offset_topic,
                10,
            )
        # 차선 검출 진단 로깅용 프레임 카운터(매 프레임 대신 스로틀).
        self._lane_log_count = 0

        self.get_logger().info(
            f'OpenCV node started: subscribe_topic={subscribe_topic}, '
            f'jpeg_quality={self.jpeg_quality}, publish_lane={self.publish_lane}, '
            f'lane_offset_topic={lane_offset_topic}'
        )

    def to_compressed_msg(self, image, source_msg: CompressedImage, frame_id: str):
        ok, encoded = cv2.imencode(
            '.jpg',
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            self.get_logger().warning(f'Failed to encode image for frame_id={frame_id}')
            return None

        out_msg = CompressedImage()
        out_msg.header.stamp = source_msg.header.stamp
        out_msg.header.frame_id = frame_id
        out_msg.format = 'jpeg'
        out_msg.data = encoded.tobytes()
        return out_msg


    def image_callback(self, msg: CompressedImage):
        raw_data = np.frombuffer(msg.data, dtype=np.uint8)
        np_arr = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)

        if np_arr is None:
            self.get_logger().warning('Failed to decode compressed image')
            return

        gray = cv2.cvtColor(np_arr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blur, 50, 150)

        gray_msg = self.to_compressed_msg(gray, msg, frame_id='opencv_grayscale')
        blur_msg = self.to_compressed_msg(blur, msg, frame_id='opencv_blur')
        edge_msg = self.to_compressed_msg(edge, msg, frame_id='opencv_edge')
        if gray_msg is None or blur_msg is None or edge_msg is None:
            return

        self.gray_pub.publish(gray_msg)
        self.blur_pub.publish(blur_msg)
        self.edge_pub.publish(edge_msg)

        if self.lane_pub is not None:
            lane = compute_lane_offset(
                np_arr,
                roi_top=self.roi_top,
                roi_left=self.roi_left,
                num_bands=self.lane_num_bands,
                valid_min_px=self.lane_valid_min_px,
                polarity=self.lane_polarity,
                method=self.lane_method,
                hsv_lower=self.lane_hsv_lower,
                hsv_upper=self.lane_hsv_upper,
            )
            lane_msg = Float32MultiArray()
            lane_msg.data = [lane.offset, 1.0 if lane.valid else 0.0, lane.curvature]
            self.lane_pub.publish(lane_msg)

            # 차선 검출 진단 로깅 — 약 15프레임마다 valid/offset/curvature/픽셀수를
            # 출력한다. valid=False 인데 pixels 가 valid_min_px(기본 40) 근처면
            # HSV/threshold 범위가 라인 색과 안 맞는 것이고, pixels 가 0 이면
            # ROI 안에 검출 색이 아예 없다는 뜻(라인 색/조명/ROI 재확인).
            self._lane_log_count += 1
            if self._lane_log_count >= 15:
                self._lane_log_count = 0
                self.get_logger().info(
                    f'lane: valid={lane.valid} offset={lane.offset:+.3f} '
                    f'curvature={lane.curvature:+.3f} pixels={lane.pixels} '
                    f'(method={self.lane_method}, valid_min_px={self.lane_valid_min_px})'
                )

        if self.debug_log:
            self.get_logger().info('Published grayscale/blur/edge frames')

    

def main(args=None):
    rclpy.init(args=args)
    node = OpenCvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
