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
        # [offset, valid, curvature] 로 발행. BEV·슬라이딩 윈도우를 실트랙에서
        # 튜닝할 수 있도록 ROS param 으로 노출 (CLAUDE.md 9.4).
        self.declare_parameter('publish_lane', True)
        self.declare_parameter('lane_offset_topic', '/lane/offset')
        # BEV 슬라이딩 윈도우 오버레이 디버그 영상(모니터에서 튜닝 확인용).
        self.declare_parameter('publish_lane_debug', True)
        self.declare_parameter('lane_debug_topic', '/opencv/image/lane')
        self.declare_parameter('lane_valid_min_px', 40)

        # --- 트랙 프로파일 (CLAUDE.md 10번) ---
        # 검출 특성이 반대인 두 트랙을 프리셋으로 보존. 기본은 대회 규정 트랙.
        #   white_track (기본, 대회): 검은 바닥 + 양쪽 흰 경계선
        #       → method='white', polarity='light' (채도 게이팅 흰색 마스크)
        #   orange_track (연습): 회색 바닥 + 주황 라인
        #       → method='color', polarity='dark'
        # 아래 개별 param 을 '명시'하면 프리셋을 덮어쓴다(개별 param 우선).
        # 명시 안 함을 나타내는 센티널: 문자열 ''.
        self.declare_parameter('lane_profile', 'white_track')
        self.declare_parameter('lane_method', '')     # '' → 프로파일 프리셋
        self.declare_parameter('lane_polarity', '')   # '' → 프로파일 프리셋
        # 주황 라인 기본 HSV 범위(OpenCV H 0~180). 실측: 주황≈H5~22.
        # (method 가 'color' 로 해석될 때만 사용 — white_track 은 무시)
        self.declare_parameter('lane_hsv_lower', [5, 80, 80])
        self.declare_parameter('lane_hsv_upper', [22, 255, 255])
        # 흰색 마스크(method='white', 대회 white_track 기본) 게이팅.
        # inRange(hsv, (0,0,v_min), (180,s_max,255)) — 저채도·고명도만 흰색으로 통과.
        # s_max 낮을수록 유채색(파란 매트) 배제 강함, v_min 높을수록 밝은 것만 통과.
        # 기본값은 실트랙 bag HSV 실측 기준(흰 선 S≤12/V≥215, 파란 매트 S≥92,
        # 노면 V≤118). s_max=50 은 [12,92], v_min=150 은 [118,215] 사이 마진값.
        self.declare_parameter('lane_white_s_max', 50)
        self.declare_parameter('lane_white_v_min', 150)
        # True 면 흰색 마스크와 명암(brightness) 마스크를 AND 결합(기본 순수 흰색 단독).
        self.declare_parameter('lane_white_combine', False)
        # 디노이즈 커널(형태학적 열림). <=1 이면 비활성.
        self.declare_parameter('morph_ksize', 3)
        # 적응형 임계값 이웃 창(brightness 경로). 해상도에 비례해 스케일할 것.
        self.declare_parameter('lane_block_size', 25)
        self.declare_parameter('blur_ksize', 5)

        # --- BEV 원근변환 4점(원본 폭/높이 대비 0~1 비율, 좌상→우상→우하→좌하) ---
        # 실트랙 bag(track_full_20260714_082403)의 직선·중앙 구간(프레임 1380±)에서
        # 차선 경계선을 추적·적합해 캘리브레이션한 값(대칭, 경계선을 BEV 15/85% 열에
        # 배치 → 직선 차선이 조감도에서 세로 평행선, offset=0=카메라 중심축).
        # ⚠️ 카메라 장착 위치/각도가 바뀌면 재캘리브레이션 필요(tools 로 재산출).
        self.declare_parameter('bev_src_tl', [0.234, 0.62])
        self.declare_parameter('bev_src_tr', [0.766, 0.62])
        self.declare_parameter('bev_src_br', [0.982, 1.00])
        self.declare_parameter('bev_src_bl', [0.018, 1.00])
        self.declare_parameter('bev_warp_w', 200)
        self.declare_parameter('bev_warp_h', 240)

        # --- 슬라이딩 윈도우 & 유효성 ---
        self.declare_parameter('n_windows', 10)
        self.declare_parameter('margin', 30)          # 윈도우 반너비(px, BEV)
        self.declare_parameter('minpix', 25)          # 윈도우 재중심화 최소 픽셀
        self.declare_parameter('hist_ratio', 0.5)     # 히스토그램에 쓸 하단 비율
        self.declare_parameter('min_lane_px', 200)    # 한쪽 라인 인정 최소 누적 픽셀
        self.declare_parameter('lane_width_ratio', 0.55)  # 한쪽 소실 폴백 차폭(BEV 폭 비율)

        subscribe_topic = str(self.get_parameter('subscribe_topic').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.debug_log = bool(self.get_parameter('debug_log').value)

        self.publish_lane = bool(self.get_parameter('publish_lane').value)
        lane_offset_topic = str(self.get_parameter('lane_offset_topic').value)
        self.publish_lane_debug = bool(self.get_parameter('publish_lane_debug').value)
        lane_debug_topic = str(self.get_parameter('lane_debug_topic').value)
        self.lane_valid_min_px = int(self.get_parameter('lane_valid_min_px').value)
        self.lane_hsv_lower = [int(v) for v in self.get_parameter('lane_hsv_lower').value]
        self.lane_hsv_upper = [int(v) for v in self.get_parameter('lane_hsv_upper').value]
        self.lane_white_s_max = int(self.get_parameter('lane_white_s_max').value)
        self.lane_white_v_min = int(self.get_parameter('lane_white_v_min').value)
        self.lane_white_combine = bool(self.get_parameter('lane_white_combine').value)
        self.morph_ksize = int(self.get_parameter('morph_ksize').value)
        self.lane_block_size = int(self.get_parameter('lane_block_size').value)
        self.blur_ksize = int(self.get_parameter('blur_ksize').value)

        self.bev_src_tl = [float(v) for v in self.get_parameter('bev_src_tl').value]
        self.bev_src_tr = [float(v) for v in self.get_parameter('bev_src_tr').value]
        self.bev_src_br = [float(v) for v in self.get_parameter('bev_src_br').value]
        self.bev_src_bl = [float(v) for v in self.get_parameter('bev_src_bl').value]
        self.bev_warp_w = int(self.get_parameter('bev_warp_w').value)
        self.bev_warp_h = int(self.get_parameter('bev_warp_h').value)

        self.n_windows = int(self.get_parameter('n_windows').value)
        self.margin = int(self.get_parameter('margin').value)
        self.minpix = int(self.get_parameter('minpix').value)
        self.hist_ratio = float(self.get_parameter('hist_ratio').value)
        self.min_lane_px = int(self.get_parameter('min_lane_px').value)
        self.lane_width_ratio = float(self.get_parameter('lane_width_ratio').value)

        # --- 프로파일 프리셋 해석 (개별 param 명시 시 덮어씀) ---
        profiles = {
            'white_track': {'method': 'white', 'polarity': 'light'},
            'orange_track': {'method': 'color', 'polarity': 'dark'},
        }
        self.lane_profile = str(self.get_parameter('lane_profile').value)
        preset = profiles.get(self.lane_profile, profiles['white_track'])

        method_p = str(self.get_parameter('lane_method').value).strip()
        self.lane_method = method_p if method_p else preset['method']
        polarity_p = str(self.get_parameter('lane_polarity').value).strip()
        self.lane_polarity = polarity_p if polarity_p else preset['polarity']

        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.subscription = self.create_subscription(
            CompressedImage, subscribe_topic, self.image_callback, image_qos)

        # monitor 대시보드가 구독하는 기존 디버그 토픽 유지.
        self.gray_pub = self.create_publisher(CompressedImage, '/opencv/image/grayscale', image_qos)
        self.blur_pub = self.create_publisher(CompressedImage, '/opencv/image/blur', image_qos)
        self.edge_pub = self.create_publisher(CompressedImage, '/opencv/image/edge', image_qos)

        self.lane_pub = None
        if self.publish_lane:
            self.lane_pub = self.create_publisher(Float32MultiArray, lane_offset_topic, 10)
        self.lane_debug_pub = None
        if self.publish_lane_debug:
            self.lane_debug_pub = self.create_publisher(CompressedImage, lane_debug_topic, image_qos)

        self._lane_log_count = 0

        self.get_logger().info(
            f'OpenCV node started (BEV+sliding-window): subscribe_topic={subscribe_topic}, '
            f'publish_lane={self.publish_lane}, lane_offset_topic={lane_offset_topic}, '
            f'lane_profile={self.lane_profile} (method={self.lane_method}, '
            f'polarity={self.lane_polarity}), warp={self.bev_warp_w}x{self.bev_warp_h}, '
            f'n_windows={self.n_windows}, lane_debug={self.publish_lane_debug}'
        )

    def to_compressed_msg(self, image, source_msg: CompressedImage, frame_id: str):
        ok, encoded = cv2.imencode(
            '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
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
            want_overlay = self.lane_debug_pub is not None
            lane = compute_lane_offset(
                np_arr,
                method=self.lane_method,
                polarity=self.lane_polarity,
                hsv_lower=self.lane_hsv_lower,
                hsv_upper=self.lane_hsv_upper,
                white_s_max=self.lane_white_s_max,
                white_v_min=self.lane_white_v_min,
                white_combine=self.lane_white_combine,
                block_size=self.lane_block_size,
                blur_ksize=self.blur_ksize,
                morph_ksize=self.morph_ksize,
                bev_src_tl=self.bev_src_tl,
                bev_src_tr=self.bev_src_tr,
                bev_src_br=self.bev_src_br,
                bev_src_bl=self.bev_src_bl,
                warp_w=self.bev_warp_w,
                warp_h=self.bev_warp_h,
                n_windows=self.n_windows,
                margin=self.margin,
                minpix=self.minpix,
                hist_ratio=self.hist_ratio,
                min_lane_px=self.min_lane_px,
                lane_width_ratio=self.lane_width_ratio,
                valid_min_px=self.lane_valid_min_px,
                draw=want_overlay,
            )
            # 발행 계약은 [offset, valid, curvature] 3원소 유지(문서화된 인터페이스).
            lane_msg = Float32MultiArray()
            lane_msg.data = [lane.offset, 1.0 if lane.valid else 0.0, lane.curvature]
            self.lane_pub.publish(lane_msg)

            if want_overlay and lane.overlay is not None:
                dbg = self.to_compressed_msg(lane.overlay, msg, frame_id='opencv_lane')
                if dbg is not None:
                    self.lane_debug_pub.publish(dbg)

            # 진단 로깅 — 약 15프레임마다. valid=False 인데 pixels 가 valid_min_px
            # 근처면 BEV 안에 라인이 거의 없다는 뜻(BEV 4점/HSV/threshold 재확인).
            self._lane_log_count += 1
            if self._lane_log_count >= 15:
                self._lane_log_count = 0
                self.get_logger().info(
                    f'lane: valid={lane.valid} offset={lane.offset:+.3f} '
                    f'curvature={lane.curvature:+.3f} pixels={lane.pixels} '
                    f'mask_px={lane.mask_pixels} '
                    f'valid_bands={lane.valid_bands} L={lane.left_detected} '
                    f'R={lane.right_detected} '
                    f'(profile={self.lane_profile}, method={self.lane_method})'
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
