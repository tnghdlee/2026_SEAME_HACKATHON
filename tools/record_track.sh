#!/usr/bin/env bash
# 실트랙 녹화 + 20Hz 조향 수정 검증.
#
# auto_driving 스택을 안전 모드(require_green_start:=false, cruise_throttle:=0.0 →
# 바퀴는 안 굴러가고 조향 서보만 동작)로 띄우고, /control 레이트를 먼저 찍어
# 20Hz 수정을 검증한 뒤, 실트랙 튜닝/진단에 필요한 토픽들을 rosbag2 로 녹화한다.
#
# 사용:
#   tools/record_track.sh            # Ctrl+C(또는 SIGINT/TERM) 로 종료할 때까지 녹화
#   tools/record_track.sh 120        # 120초 녹화 후 자동 종료
#
# 종료 시 bag 을 SIGINT 로 정상 마감(metadata.yaml 기록)한 뒤 스택을 내린다.
# 산출물: bagfile/track_full_<STAMP>/ (+ launch 로그).
# ROS setup.bash 는 -u(unbound) 비호환이라 -u 는 쓰지 않는다.
set -o pipefail

ROOT=/home/topst/D-Racer-Kit
DURATION="${1:-0}"          # 초. 0 = 무한(신호로 종료).

source /opt/ros/humble/setup.bash
source "$ROOT/install/setup.bash"

STAMP=$(date +%Y%m%d_%H%M%S)
BAG="$ROOT/bagfile/track_full_$STAMP"
LOG="$ROOT/bagfile/launch_$STAMP.log"

STOPPED=0
cleanup() {
  [[ "$STOPPED" == "1" ]] && return
  STOPPED=1
  echo "[record_track] 종료 — bag 마감 중..."
  [[ -n "${BAG_PID:-}" ]] && kill -INT "$BAG_PID" 2>/dev/null || true
  sleep 3
  [[ -n "${LAUNCH_PID:-}" ]] && kill -INT "$LAUNCH_PID" 2>/dev/null || true
  sleep 3
  echo "[record_track] 완료. bag: $BAG"
}
trap cleanup INT TERM

echo "[record_track] auto_driving 실행(throttle 0 → 바퀴 정지, 조향만). log=$LOG"
ros2 launch control auto_driving.launch.py \
  require_green_start:=false cruise_throttle:=0.0 >"$LOG" 2>&1 &
LAUNCH_PID=$!

echo "[record_track] 토픽 대기(/camera/image/compressed, /control)..."
for _ in $(seq 1 40); do
  t=$(ros2 topic list 2>/dev/null || true)
  if grep -q '/camera/image/compressed' <<<"$t" && grep -q '/control' <<<"$t"; then
    echo "[record_track] 토픽 준비됨."
    break
  fi
  sleep 1
done

echo "[record_track] /control 레이트 확인(20Hz 수정 검증, ~6s)..."
timeout 6 ros2 topic hz /control 2>&1 | tail -4 || true

echo "[record_track] 녹화 시작 -> $BAG"
ros2 bag record -o "$BAG" \
  /camera/image/compressed /lane/offset /control /inference/detections /opencv/image/lane \
  >/dev/null 2>&1 &
BAG_PID=$!

if [[ "$DURATION" -gt 0 ]]; then
  echo "[record_track] ${DURATION}s 후 자동 종료(또는 신호). 지금 트랙을 주행/이동하세요."
  sleep "$DURATION"
  cleanup
else
  echo "[record_track] 녹화 중. 트랙을 한 바퀴 돌린 뒤 이 프로세스에 SIGINT/SIGTERM 을 보내 종료."
  while [[ "$STOPPED" == "0" ]]; do sleep 1; done
fi
