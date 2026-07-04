#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lane_follower_node.py

D-Racer-Kit 차선 추종 노드 (CLAUDE.md 8.1 "inference 노드 누락" 해결용 스타터).

역할:
  /camera/image/compressed (sensor_msgs/CompressedImage)  구독
    -> ROI 크롭 -> 밝기(HLS) + Canny 결합 이진화
    -> 여러 스캔 행에서 좌/우 차선 x 추정 -> 중심선/오차 계산
    -> PD 제어로 steering 산출
  /control (control_msgs/Control)  발행  ->  control_node 가 모터/서보 구동

설계 원칙(CLAUDE.md 9번) 반영:
  - 완주 안정성 > 속도: 미검출 시 급조향 대신 감속/직진 유지
  - 방향 파라미터화: direction 파라미터로 정/역(좌우대칭) 미러링
  - 시간적 필터링: 연속 N프레임 미검출이어야 '차선 상실'로 판단

주의(꼭 확인):
  - control_msgs/Control 의 steering/throttle 범위·부호 규약은 control_node.py 기준으로 맞추세요.
    아래는 steering, throttle 모두 [-1.0, 1.0] 정규화, steering>0 = 우회전 가정입니다.
  - auto_driving.launch.py 가 기대하는 실행 노드 이름(예: inference_node)과 맞추거나,
    런치의 executable 명을 이 노드에 맞게 수정하세요. (8.1)
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from control_msgs.msg import Control


class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')

        # ---- 파라미터 (튜닝 지점) ----
        self.declare_parameter('subscribe_topic', '/camera/image/compressed')
        self.declare_parameter('publish_topic', '/control')

        self.declare_parameter('image_width', 320)
        self.declare_parameter('image_height', 160)
        self.declare_parameter('roi_top', 50)        # 이 y 위쪽은 버림 (vehicle_config ROI_TOP과 맞추기)

        # 정방향 'forward' / 역방향 'reverse' (좌우 대칭이므로 조향 부호만 반전)
        self.declare_parameter('direction', 'forward')

        # 이진화 임계값 (밝은 차선 기준)
        self.declare_parameter('bright_thresh', 160)  # HLS L 채널 임계값 (조명 따라 140~200)
        self.declare_parameter('canny_low', 50)
        self.declare_parameter('canny_high', 150)

        # 차선 폭(픽셀) — 한쪽만 보일 때 반대쪽 추정에 사용
        self.declare_parameter('lane_width_px', 200)

        # 제어 게인 / 출력
        self.declare_parameter('kp', 0.006)          # cross-track error 게인
        self.declare_parameter('kd', 0.003)          # heading error 게인
        self.declare_parameter('max_steer', 1.0)
        self.declare_parameter('base_throttle', 0.25)  # 저속에서 시작해 올리세요
        self.declare_parameter('curve_slowdown', 0.5)   # 큰 조향 시 throttle 감쇠 비율

        # 시간적 필터링
        self.declare_parameter('lost_frames_limit', 5)  # 연속 미검출 N프레임 -> 차선 상실

        self.declare_parameter('num_scan_rows', 8)     # ROI를 몇 개 행으로 스캔할지
        self.declare_parameter('debug_log', False)

        # 디버그 시각화 (튜닝용): 검출 결과를 그려 별도 토픽으로 발행
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('debug_topic', '/lane_follower/debug/compressed')

        gp = self.get_parameter
        self.sub_topic = gp('subscribe_topic').value
        self.pub_topic = gp('publish_topic').value
        self.W = int(gp('image_width').value)
        self.H = int(gp('image_height').value)
        self.roi_top = int(gp('roi_top').value)
        self.dir_sign = -1.0 if gp('direction').value == 'reverse' else 1.0
        self.bright_thresh = int(gp('bright_thresh').value)
        self.canny_low = int(gp('canny_low').value)
        self.canny_high = int(gp('canny_high').value)
        self.lane_width = int(gp('lane_width_px').value)
        self.kp = float(gp('kp').value)
        self.kd = float(gp('kd').value)
        self.max_steer = float(gp('max_steer').value)
        self.base_throttle = float(gp('base_throttle').value)
        self.curve_slowdown = float(gp('curve_slowdown').value)
        self.lost_limit = int(gp('lost_frames_limit').value)
        self.num_rows = int(gp('num_scan_rows').value)
        self.debug_log = bool(gp('debug_log').value)
        self.publish_debug = bool(gp('publish_debug').value)
        self.debug_topic = gp('debug_topic').value

        # ---- 상태 ----
        self.prev_center = self.W / 2.0
        self.prev_cte = 0.0
        self.lost_count = 0
        self.last_scan = []   # 디버그용: [(y, left_x, right_x, center_x), ...]

        self.pub = self.create_publisher(Control, self.pub_topic, 10)
        self.sub = self.create_subscription(
            CompressedImage, self.sub_topic, self.on_image, 10)
        self.debug_pub = None
        if self.publish_debug:
            self.debug_pub = self.create_publisher(
                CompressedImage, self.debug_topic, 10)

        self.get_logger().info(
            f'[lane_follower] sub={self.sub_topic} pub={self.pub_topic} '
            f'dir_sign={self.dir_sign} size={self.W}x{self.H}')

    # -------------------------------------------------------------------------
    def on_image(self, msg: CompressedImage):
        frame = self.decode(msg)
        if frame is None:
            return

        binary = self.to_binary(frame)               # ROI 이진화
        center_x, found = self.find_lane_center(binary)

        if not found:
            self.lost_count += 1
        else:
            self.lost_count = 0
            self.prev_center = center_x

        # ROI 좌표계 기준 화면 중앙과의 오차
        cte = self.prev_center - (self.W / 2.0)       # +면 차선 중심이 오른쪽 -> 우조향 필요
        heading = cte - self.prev_cte                  # 단순 미분(변화량)
        self.prev_cte = cte

        steering = self.kp * cte + self.kd * heading
        steering *= self.dir_sign                      # 역방향 미러링
        steering = float(np.clip(steering, -self.max_steer, self.max_steer))

        # 큰 조향일수록 감속 (곡선 안정성)
        throttle = self.base_throttle * (1.0 - self.curve_slowdown * abs(steering))

        # 차선 상실 시: 급조향 금지, 직전 방향 약하게 유지 + 감속
        if self.lost_count >= self.lost_limit:
            steering = float(np.clip(steering * 0.3, -self.max_steer, self.max_steer))
            throttle = self.base_throttle * 0.4
            if self.debug_log:
                self.get_logger().warn(f'lane lost ({self.lost_count}) -> slow & hold')

        self.publish_control(steering, throttle, msg.header)

        if self.debug_pub is not None:
            self.publish_debug_image(frame, binary, self.prev_center, steering, found)

        if self.debug_log:
            self.get_logger().info(
                f'cte={cte:+.1f} steer={steering:+.3f} thr={throttle:.3f} found={found}')

    # -------------------------------------------------------------------------
    def decode(self, msg: CompressedImage):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            # 카메라 해상도와 파라미터가 다르면 맞춰줌
            if img.shape[1] != self.W or img.shape[0] != self.H:
                img = cv2.resize(img, (self.W, self.H))
            return img
        except Exception as e:
            self.get_logger().error(f'decode failed: {e}')
            return None

    def to_binary(self, frame):
        """ROI만 남기고 밝은 차선(HLS L) + 엣지(Canny)를 결합해 이진화."""
        roi = frame[self.roi_top:self.H, 0:self.W]

        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        _, bright = cv2.threshold(l_channel, self.bright_thresh, 255, cv2.THRESH_BINARY)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blur, self.canny_low, self.canny_high)

        combined = cv2.bitwise_or(bright, edge)
        return combined  # shape: (H-roi_top, W)

    def find_lane_center(self, binary):
        """
        여러 스캔 행에서 좌/우 차선 x를 찾아 중심을 추정.
        - 가장 아래 행부터 위로 num_rows개 밴드로 스캔
        - 이전 중심 기준 좌/우로 나눠 각 영역의 흰 픽셀 무게중심(x) 계산
        - 한쪽만 있으면 lane_width/2 오프셋으로 반대쪽 추정
        반환: (center_x[전체영상 좌표], found)
        """
        h = binary.shape[0]
        band = max(1, h // self.num_rows)
        mid = self.prev_center  # 이전 프레임 중심 기준으로 좌/우 분할

        centers = []
        self.last_scan = []   # 디버그 초기화 (좌표는 ROI 기준)
        for i in range(self.num_rows):
            y0 = h - (i + 1) * band
            y1 = h - i * band
            if y0 < 0:
                y0 = 0
            row = binary[y0:y1, :]
            colsum = np.sum(row, axis=0).astype(np.float32)  # 각 x열 흰 픽셀량

            left_region = colsum[:int(mid)]
            right_region = colsum[int(mid):]

            left_x = self._weighted_x(left_region, offset=0)
            right_x = self._weighted_x(right_region, offset=int(mid))

            if left_x is not None and right_x is not None:
                c = (left_x + right_x) / 2.0
            elif left_x is not None:
                c = left_x + self.lane_width / 2.0
            elif right_x is not None:
                c = right_x - self.lane_width / 2.0
            else:
                continue
            centers.append(c)
            self.last_scan.append(((y0 + y1) // 2, left_x, right_x, c))

        if not centers:
            return self.prev_center, False

        # 아래쪽(최근) 밴드에 가중치를 더 주어 안정화
        center_x = float(np.mean(centers))
        center_x = float(np.clip(center_x, 0, self.W))
        return center_x, True

    @staticmethod
    def _weighted_x(region, offset):
        total = np.sum(region)
        if total < 255 * 2:  # 최소 흰 픽셀량 미달이면 없음 처리
            return None
        xs = np.arange(region.shape[0], dtype=np.float32)
        return float(np.sum(xs * region) / total) + offset

    def publish_control(self, steering, throttle, header):
        msg = Control()
        msg.header = header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.steering = float(steering)
        msg.throttle = float(throttle)
        self.pub.publish(msg)

    def publish_debug_image(self, frame, binary, center_x, steering, found):
        """
        튜닝용 시각화:
          - 이진화(binary)를 초록 채널에 얹어 '검출된 픽셀'을 눈으로 확인
          - 스캔 행별 좌(파랑)/우(빨강) 차선점, 추정 중심(노랑)
          - 화면 중앙선(흰색) 대비 차선 중심 위치
          - 좌상단에 steering / found 텍스트
        rqt_image_view 로 debug_topic 을 열어 보면서 임계값을 맞추세요.
        """
        try:
            vis = frame.copy()
            roi_off = self.roi_top

            # 이진화 결과를 초록으로 오버레이 (ROI 영역만)
            mask_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            mask_bgr[:, :, 0] = 0
            mask_bgr[:, :, 2] = 0  # 초록만 남김
            vis[roi_off:self.H, 0:self.W] = cv2.addWeighted(
                vis[roi_off:self.H, 0:self.W], 0.6, mask_bgr, 0.4, 0)

            # 화면 중앙선 (흰색)
            cv2.line(vis, (self.W // 2, roi_off), (self.W // 2, self.H), (255, 255, 255), 1)

            # 스캔 행별 좌/우/중심 표시 (좌표는 ROI 기준 -> roi_off 더함)
            for (y, lx, rx, c) in self.last_scan:
                yy = int(y) + roi_off
                if lx is not None:
                    cv2.circle(vis, (int(lx), yy), 2, (255, 0, 0), -1)   # 좌: 파랑
                if rx is not None:
                    cv2.circle(vis, (int(rx), yy), 2, (0, 0, 255), -1)   # 우: 빨강
                cv2.circle(vis, (int(c), yy), 2, (0, 255, 255), -1)      # 중심: 노랑

            # 최종 추정 중심 세로선 (노랑)
            cv2.line(vis, (int(center_x), roi_off), (int(center_x), self.H), (0, 255, 255), 1)

            cv2.putText(vis, f'steer={steering:+.2f} found={int(found)}',
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            ok, enc = cv2.imencode('.jpg', vis)
            if not ok:
                return
            out = CompressedImage()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = 'lane_follower_debug'
            out.format = 'jpeg'
            out.data = enc.tobytes()
            self.debug_pub.publish(out)
        except Exception as e:
            self.get_logger().warn(f'debug image failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()