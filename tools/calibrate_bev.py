#!/usr/bin/env python3
"""BEV 4점 캘리브레이션 산출기 (rosbag2 카메라 프레임 기반).

카메라 장착 위치/각도가 바뀌면 BEV 사다리꼴(bev_src_*)을 다시 맞춰야 한다. 이 도구는
실주행 bag 에서 **직선·중앙** 프레임을 자동으로 찾아 차선 경계선을 추적·적합하고,
직선 차선이 조감도에서 세로 평행선이 되도록(=offset 0 이 카메라 중심축) 대칭 사다리꼴
4점을 계산해 출력한다. 흰색 마스크 임계는 lane_detect 기본값(white_track)을 쓴다.

사용:
    python3 tools/calibrate_bev.py bagfile/track_full_YYYYMMDD_HHMMSS
    # → bev_src_tl/tr/br/bl 출력 + calib_bev.jpg(조감도 오버레이) 저장.
출력된 4점을 opencv_node 의 declare_parameter 기본값(또는 launch -p)에 반영한다.

의존: cv2, numpy (ROS 불필요). src/opencv 가 PYTHONPATH 에 있어야 lane_detect import 가능.
"""
import glob
import json
import os
import sqlite3
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'opencv'))
from opencv.lane_detect import compute_lane_offset, _white_mask  # noqa: E402

# 캘리브레이션 대상 파라미터(대회 white_track 실효값). 필요시 수정.
S_MAX, V_MIN = 50, 150
BLOCK, MORPH = 63, 7
Y_TOP, Y_BOT = 0.62, 1.00     # 사다리꼴 상/하단 높이 비율
EDGE_FRAC = 0.15              # 경계선을 BEV 폭의 이 비율 위치에 배치(15%/85%)
BASE = dict(method='white', polarity='light', white_s_max=S_MAX, white_v_min=V_MIN,
            block_size=BLOCK, blur_ksize=5, morph_ksize=MORPH, warp_w=200, warp_h=240,
            n_windows=10, margin=30, minpix=25, hist_ratio=0.5, min_lane_px=200,
            lane_width_ratio=0.55, valid_min_px=250)


def load_frames(bag_dir):
    db = sorted(glob.glob(os.path.join(bag_dir, '*.db3')))[0]
    con = sqlite3.connect(db)
    tid = [r[0] for r in con.execute(
        "SELECT id FROM topics WHERE name='/camera/image/compressed'")][0]
    blobs = [r[0] for r in con.execute(
        'SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))]
    con.close()
    return blobs


def decode(blob):
    b = bytes(blob)
    s = b.find(b'\xff\xd8\xff')
    e = b.rfind(b'\xff\xd9')
    return cv2.imdecode(np.frombuffer(b[s:e + 2], np.uint8), cv2.IMREAD_COLOR)


def find_straight_centered(blobs, n=80):
    """|offset|+2|curvature| 가 가장 작은(직선+중앙) 프레임 인덱스들을 반환."""
    idxs = np.linspace(0, len(blobs) - 1, n).astype(int)
    scored = []
    for i in idxs:
        r = compute_lane_offset(decode(blobs[i]), **BASE)
        if r.valid and r.valid_bands == 2:
            scored.append((abs(r.offset) + 2 * abs(r.curvature), int(i)))
    scored.sort()
    return [i for _, i in scored[:6]]


def trace_borders(blobs, cal_idxs):
    """캘리브레이션 프레임들에서 차선 경계선 점을 모아 직선 적합."""
    LP, RP = [], []
    for i in cal_idxs:
        img = decode(blobs[i])
        h, w = img.shape[:2]
        c = w // 2
        m = _white_mask(img, S_MAX, V_MIN, 5)
        for y in range(int(0.63 * h), int(0.97 * h), 3):
            row = m[y]
            if row[c] > 0:
                continue
            ls = np.where(row[:c] > 0)[0]
            rs = np.where(row[c:] > 0)[0]
            if len(ls) and len(rs):
                xL, xR = ls[-1], rs[0] + c
                if 0.15 * w < (xR - xL) < 0.9 * w:
                    LP.append((y, xL))
                    RP.append((y, xR))
    return np.array(LP, float), np.array(RP, float)


def compute_trapezoid(LP, RP, w, h):
    mL, bL = np.polyfit(LP[:, 0], LP[:, 1], 1)
    mR, bR = np.polyfit(RP[:, 0], RP[:, 1], 1)
    c = w / 2.0
    k = EDGE_FRAC / (1.0 - 2 * EDGE_FRAC)
    yb, yt = Y_BOT * h, Y_TOP * h

    def outer(y):
        left, right = mL * y + bL, mR * y + bR
        wdt = right - left
        return left - k * wdt, right + k * wdt

    xLb, xRb = outer(yb)
    xLt, xRt = outer(yt)
    halfb, halft = (xRb - xLb) / 2, (xRt - xLt) / 2   # 대칭(중심 c)
    return {
        'tl': [round(float(c - halft) / w, 3), Y_TOP],
        'tr': [round(float(c + halft) / w, 3), Y_TOP],
        'br': [round(float(c + halfb) / w, 3), Y_BOT],
        'bl': [round(float(c - halfb) / w, 3), Y_BOT],
    }


def main():
    if len(sys.argv) < 2:
        print('usage: python3 tools/calibrate_bev.py <bag_dir>')
        sys.exit(1)
    bag = sys.argv[1]
    blobs = load_frames(bag)
    print(f'frames: {len(blobs)}')
    cal = find_straight_centered(blobs)
    if not cal:
        print('직선+중앙 프레임을 찾지 못함 — 다른 bag/구간 필요.')
        sys.exit(2)
    print(f'calibration frames (straightest+centered): {cal}')
    LP, RP = trace_borders(blobs, cal)
    h, w = decode(blobs[cal[0]]).shape[:2]
    tz = compute_trapezoid(LP, RP, w, h)
    print('\n=== 캘리브레이션 결과 (opencv_node declare_parameter / launch -p 에 반영) ===')
    for key in ('tl', 'tr', 'br', 'bl'):
        print(f'  bev_src_{key} = {tz[key]}')
    # 검증 오버레이 저장
    r = compute_lane_offset(
        decode(blobs[cal[0]]), draw=True, **BASE,
        bev_src_tl=tuple(tz['tl']), bev_src_tr=tuple(tz['tr']),
        bev_src_br=tuple(tz['br']), bev_src_bl=tuple(tz['bl']))
    out = os.path.join(bag, 'calib_bev.jpg')
    cv2.imwrite(out, r.overlay)
    print(f'\nframe {cal[0]}: offset={r.offset:+.3f} curv={r.curvature:+.3f} '
          f'bands={r.valid_bands} (직선 프레임이라 offset≈0 이어야 정상)')
    print(f'조감도 검증 영상 저장: {out} (좌우 경계선이 세로 평행선이면 OK)')
    print(json.dumps(tz))


if __name__ == '__main__':
    main()
