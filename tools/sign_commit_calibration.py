#!/usr/bin/env python3
"""갈림길 표지판 커밋 임계(sign_commit_ratio) 캘리브레이션 도구 (read-only).

/inference/detections (std_msgs/String, JSON) 를 구독해, 방향 표지판(left_sign/
right_sign)이 잡힐 때마다 driving_policy._sign_proximity 와 '완전히 동일한' 근접
지표 3종(bottom_y / area / height)을 실시간 출력한다. inference_node 코드는 건드리지
않는다.

계산식 (driving_policy.py:_sign_proximity 와 1:1):
  bottom_y = y2 / H
  area     = (x2-x1)*(y2-y1) / (W*H)
  height   = (y2-y1) / H

사용:
  # cruise_throttle:=0.0 로 auto_driving 를 띄운 상태(바퀴 안 굴러감, 인식만)에서:
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 tools/sign_commit_calibration.py
  # 차를 갈림길로 천천히 밀며, '지금 꺾어야 한다' 싶은 지점의 bottom_y 를 여러 번 읽어 평균.
  # 그 평균을 그대로:  ros2 launch control auto_driving.launch.py sign_commit_ratio:=<평균>

  # 샘플 자동 집계 모드(엔터 없이, 최근 N프레임 평균/최댓값을 계속 표시):
  python3 tools/sign_commit_calibration.py --window 23

'현재 launch 의 임계/지표'와 비교하려면 인자로 넘겨 화면에 표시(판정 미리보기):
  python3 tools/sign_commit_calibration.py --metric bottom_y --ratio 0.62
"""
import argparse
import json
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

CLASS = {0: 'redlight', 1: 'greenlight', 2: 'left_sign', 3: 'right_sign'}
SIGN_IDS = (2, 3)


class Calib(Node):
    def __init__(self, metric, ratio, window):
        super().__init__('sign_commit_calibration')
        self.metric = metric
        self.ratio = ratio
        self.window = window
        self.samples = deque(maxlen=window)  # 선택 지표 최근값(집계용)
        self.create_subscription(String, '/inference/detections', self.cb, 10)
        self.get_logger().info(
            f'sign_commit_calibration 시작. 비교기준 metric={metric} '
            f'ratio={ratio} (판정 미리보기). window={window}. '
            f'표지판을 카메라에 두고 차를 천천히 미세요. Ctrl-C 종료.')

    def cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        dets = payload.get('detections', [])
        hw = payload.get('frame_hw')
        if not hw or len(hw) != 2:
            return
        H, W = float(hw[0]), float(hw[1])
        if H <= 0 or W <= 0:
            return
        signs = [d for d in dets if int(d.get('cls', -1)) in SIGN_IDS]
        if not signs:
            return
        # 가장 가까운(=지표 큰) 표지판 하나를 대표로 (여러 개면 최댓값)
        best = None
        best_val = -1.0
        for d in signs:
            x1, y1, x2, y2 = d['box']
            m = self._metrics(x1, y1, x2, y2, H, W)
            v = m[self.metric]
            if v > best_val:
                best_val = v
                best = (d, m)
        d, m = best
        self.samples.append(m[self.metric])
        cls = CLASS.get(int(d['cls']), str(d['cls']))
        commit = 'COMMIT(꺾기시작)' if best_val >= self.ratio else '대기'
        avg = sum(self.samples) / len(self.samples)
        mx = max(self.samples)
        self.get_logger().info(
            f'{cls} score={d.get("score",0):.2f} | '
            f'bottom_y={m["bottom_y"]:.3f} area={m["area"]:.3f} '
            f'height={m["height"]:.3f} || [{self.metric}]={best_val:.3f} '
            f'{commit} (≥{self.ratio}) | 최근{len(self.samples)}: 평균={avg:.3f} 최대={mx:.3f}')

    @staticmethod
    def _metrics(x1, y1, x2, y2, H, W):
        return {
            'bottom_y': y2 / H,
            'area': ((x2 - x1) * (y2 - y1)) / (W * H),
            'height': (y2 - y1) / H,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', default='bottom_y',
                    choices=['bottom_y', 'area', 'height'],
                    help='판정 미리보기에 쓸 지표(정책 sign_proximity_metric 과 맞추기)')
    ap.add_argument('--ratio', type=float, default=0.60,
                    help='판정 미리보기 임계(정책 sign_commit_ratio 과 맞추기)')
    ap.add_argument('--window', type=int, default=23,
                    help='최근 N개 평균/최대 집계 창(기본 23)')
    args = ap.parse_args()
    rclpy.init()
    node = Calib(args.metric, args.ratio, args.window)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
