#!/usr/bin/env python3
"""ArUco 실차 진단기 (read-only).

/camera/image/compressed 를 구독해, 매 프레임을 '모든 주요 ArUco 사전'으로
검출하고 잡히는 마커의 (사전, ID, 면적비, 중심 정규화좌표, 기본 ROI 통과 여부)를
출력한다. inference_node 코드를 전혀 건드리지 않는다.

목적: "차가 마커를 봐도 안 멈춘다" 문제에서, 실제 마커가
  - 어떤 dict/ID 인지 (inference 기본값 DICT_6X6_50/ID=3 과 다르면 게이팅에서 걸러짐)
  - 화면 어디에 잡히는지 (기본 ROI [0.2,0.4,0.8,1.0] 밖이면 걸러짐)
  - 애초에 검출이 되긴 하는지 (조명/각도/블러/거리 문제)
를 눈으로 확인하기 위한 도구.

사용:
  # 카메라 노드가 떠 있는 상태에서(예: auto_driving.launch 중, 또는 camera_node 단독)
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 tools/aruco_scan.py

  # 특정 사전만 빠르게 보고 싶으면:
  python3 tools/aruco_scan.py --dicts DICT_6X6_50 DICT_4X4_50
"""
import argparse
import sys

import cv2
import cv2.aruco as aruco
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, HistoryPolicy, ReliabilityPolicy,
                       DurabilityPolicy)
from sensor_msgs.msg import CompressedImage

# inference 기본 게이팅과 동일한 기준(실차에서 걸러지는지 눈으로 비교용)
DEFAULT_ROI = (0.0, 0.0, 1.0, 0.6)   # aruco_roi_norm 기본값(가로 전체·위 60%)
DEFAULT_TARGET_ID = 3                # aruco_target_ids 기본값
DEFAULT_DICT = "DICT_6X6_50"         # aruco_dict 기본값

ALL_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_7X7_50": aruco.DICT_7X7_50,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
}


def in_roi(cn, roi):
    x0, y0, x1, y1 = roi
    x, y = cn
    return (x0 <= x <= x1) and (y0 <= y <= y1)


class Scan(Node):
    def __init__(self, dict_names):
        super().__init__('aruco_scan')
        params = aruco.DetectorParameters()
        self.detectors = {
            name: aruco.ArucoDetector(aruco.getPredefinedDictionary(ALL_DICTS[name]),
                                      params)
            for name in dict_names
        }
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(CompressedImage, '/camera/image/compressed',
                                 self.cb, qos)
        self._frames = 0
        self._seen = 0
        self.get_logger().info(
            f'aruco_scan 시작. 검사 사전 {list(self.detectors)} / '
            f'inference 기본 게이팅: dict={DEFAULT_DICT} id={DEFAULT_TARGET_ID} '
            f'roi={DEFAULT_ROI}. Ctrl-C 로 종료.')

    def cb(self, msg):
        raw = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            return
        self._frames += 1
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        area = float(h * w)
        found = []
        for name, det in self.detectors.items():
            corners, ids, _ = det.detectMarkers(gray)
            if ids is None:
                continue
            for c, i in zip(corners, ids.flatten()):
                pts = c.reshape(4, 2).astype(np.float32)
                cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                cn = (cx / w, cy / h)
                ar = float(cv2.contourArea(pts)) / area
                # inference 게이팅을 그대로 재현: dict 일치 + id==3 + ROI 안
                gated = (name == DEFAULT_DICT and int(i) == DEFAULT_TARGET_ID
                         and in_roi(cn, DEFAULT_ROI))
                found.append((name, int(i), ar, cn, in_roi(cn, DEFAULT_ROI), gated))
        if found:
            self._seen += 1
            for (name, i, ar, cn, roi_ok, gated) in found:
                self.get_logger().info(
                    f'[{self._frames:5d}] dict={name} id={i} '
                    f'area={ar*100:.2f}% center=({cn[0]:.2f},{cn[1]:.2f}) '
                    f'ROI통과={roi_ok} → inference정지트리거={"YES" if gated else "NO"}')
        elif self._frames % 30 == 0:
            self.get_logger().info(
                f'[{self._frames:5d}] 마커 미검출 (검출프레임 {self._seen}/{self._frames})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dicts', nargs='*', default=list(ALL_DICTS),
                    help='검사할 사전 이름들 (기본: 전체)')
    args = ap.parse_args()
    bad = [d for d in args.dicts if d not in ALL_DICTS]
    if bad:
        print(f'알 수 없는 사전: {bad}\n선택지: {list(ALL_DICTS)}', file=sys.stderr)
        sys.exit(2)
    rclpy.init()
    node = Scan(args.dicts)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
