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
        # roi_right: -1 = 전폭(=원본 이미지 오른쪽 끝). 비대칭 ROI 크롭용.
        self.declare_parameter('roi_right', -1)
        self.declare_parameter('lane_num_bands', 3)
        self.declare_parameter('lane_valid_min_px', 40)

        # --- 트랙 프로파일 (CLAUDE.md 10번) ---
        # 검출 특성이 반대인 두 트랙을 프리셋으로 보존. 기본은 대회 규정 트랙.
        #   white_track (기본, 대회): 검은 바닥 + 양쪽 흰 경계선
        #       → method='brightness', polarity='light', split_lanes=True
        #   orange_track (연습): 회색 바닥 + 주황 라인
        #       → method='color', polarity='dark', split_lanes=False
        # 아래 개별 param 을 '명시'하면 프리셋을 덮어쓴다(개별 param 우선).
        # 명시 안 함을 나타내는 센티널: 문자열 '' / split_lanes 'auto'.
        self.declare_parameter('lane_profile', 'white_track')
        self.declare_parameter('lane_method', '')     # '' → 프로파일 프리셋
        self.declare_parameter('lane_polarity', '')   # '' → 프로파일 프리셋
        self.declare_parameter('split_lanes', 'auto')  # 'auto'|'true'|'false'
        # 주황 라인 기본 HSV 범위(OpenCV H 0~180). 실측: 주황≈H5~22.
        # (method 가 'color' 로 해석될 때만 사용 — white_track 은 무시)
        self.declare_parameter('lane_hsv_lower', [5, 80, 80])
        self.declare_parameter('lane_hsv_upper', [22, 255, 255])
        # split 모드: 한쪽 라인만 보일 때 반대쪽 추정용 정규화 반차폭(실측 튜닝 대상).
        self.declare_parameter('lane_half_norm', 0.5)
        # 디노이즈 커널(형태학적 열림). <=1 이면 비활성.
        self.declare_parameter('morph_ksize', 3)
        # 적응형 임계값 이웃 창(brightness 경로). 해상도에 비례해 스케일할 것
        # (320×240 → 25, 800×600 → 63). 짝수/1 이하는 내부에서 홀수로 보정.
        self.declare_parameter('lane_block_size', 25)

        # --- Hough 라인 검출 파라미터 (lane_detect Hough 방식) ---
        # 이진 마스크 → Canny 에지 → HoughLinesP. 픽셀 기준 값은 해상도에 비례해
        # 스케일할 것(min_line_length/max_line_gap). min_angle_deg 로 near-수평
        # 세그먼트(정지선/노이즈)를 버린다. 실트랙 튜닝 대상.
        self.declare_parameter('hough_threshold', 30)
        self.declare_parameter('hough_min_line_length', 20)
        self.declare_parameter('hough_max_line_gap', 15)
        self.declare_parameter('hough_min_angle_deg', 25.0)
        self.declare_parameter('canny_low', 50)
        self.declare_parameter('canny_high', 150)

        subscribe_topic = str(self.get_parameter('subscribe_topic').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.debug_log = bool(self.get_parameter('debug_log').value)

        self.publish_lane = bool(self.get_parameter('publish_lane').value)
        lane_offset_topic = str(self.get_parameter('lane_offset_topic').value)
        self.roi_top = int(self.get_parameter('roi_top').value)
        self.roi_left = int(self.get_parameter('roi_left').value)
        roi_right = int(self.get_parameter('roi_right').value)
        self.roi_right = None if roi_right < 0 else roi_right
        self.lane_num_bands = int(self.get_parameter('lane_num_bands').value)
        self.lane_valid_min_px = int(self.get_parameter('lane_valid_min_px').value)
        self.lane_hsv_lower = [int(v) for v in self.get_parameter('lane_hsv_lower').value]
        self.lane_hsv_upper = [int(v) for v in self.get_parameter('lane_hsv_upper').value]
        self.lane_half_norm = float(self.get_parameter('lane_half_norm').value)
        self.morph_ksize = int(self.get_parameter('morph_ksize').value)
        self.lane_block_size = int(self.get_parameter('lane_block_size').value)
        self.hough_threshold = int(self.get_parameter('hough_threshold').value)
        self.hough_min_line_length = int(self.get_parameter('hough_min_line_length').value)
        self.hough_max_line_gap = int(self.get_parameter('hough_max_line_gap').value)
        self.hough_min_angle_deg = float(self.get_parameter('hough_min_angle_deg').value)
        self.canny_low = int(self.get_parameter('canny_low').value)
        self.canny_high = int(self.get_parameter('canny_high').value)

        # --- 프로파일 프리셋 해석 (개별 param 명시 시 덮어씀) ---
        profiles = {
            'white_track': {'method': 'brightness', 'polarity': 'light', 'split': True},
            'orange_track': {'method': 'color', 'polarity': 'dark', 'split': False},
        }
        self.lane_profile = str(self.get_parameter('lane_profile').value)
        preset = profiles.get(self.lane_profile, profiles['white_track'])

        method_p = str(self.get_parameter('lane_method').value).strip()
        self.lane_method = method_p if method_p else preset['method']

        polarity_p = str(self.get_parameter('lane_polarity').value).strip()
        self.lane_polarity = polarity_p if polarity_p else preset['polarity']

        split_p = str(self.get_parameter('split_lanes').value).strip().lower()
        if split_p in ('true', '1', 'yes', 'on'):
            self.split_lanes = True
        elif split_p in ('false', '0', 'no', 'off'):
            self.split_lanes = False
        else:  # 'auto' 또는 미인식 → 프로파일 프리셋
            self.split_lanes = preset['split']

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
            f'lane_offset_topic={lane_offset_topic}, '
            f'lane_profile={self.lane_profile} '
            f'(method={self.lane_method}, polarity={self.lane_polarity}, '
            f'split_lanes={self.split_lanes})'
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
                roi_right=self.roi_right,
                num_bands=self.lane_num_bands,
                valid_min_px=self.lane_valid_min_px,
                polarity=self.lane_polarity,
                method=self.lane_method,
                hsv_lower=self.lane_hsv_lower,
                hsv_upper=self.lane_hsv_upper,
                morph_ksize=self.morph_ksize,
                split_lanes=self.split_lanes,
                lane_half_norm=self.lane_half_norm,
                block_size=self.lane_block_size,
                hough_threshold=self.hough_threshold,
                hough_min_line_length=self.hough_min_line_length,
                hough_max_line_gap=self.hough_max_line_gap,
                hough_min_angle_deg=self.hough_min_angle_deg,
                canny_low=self.canny_low,
                canny_high=self.canny_high,
            )
            # 발행 계약은 [offset, valid, curvature] 3원소 유지(문서화된 인터페이스).
            # valid_bands 는 LaneResult 에 있고 아래 진단 로그로 노출 — inference 가
            # 신설되면 4번째 원소로 확장 가능.
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
                    f'valid_bands={lane.valid_bands} '
                    f'(profile={self.lane_profile}, method={self.lane_method}, '
                    f'split={self.split_lanes}, valid_min_px={self.lane_valid_min_px})'
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
