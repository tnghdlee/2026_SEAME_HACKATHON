# CLAUDE.md

이 문서는 Claude Code가 본 프로젝트를 이해하고 작업할 때 항상 참고하는 가이드입니다.
**자율주행 해커톤의 미션·규정**과 **현재 `src` 워크스페이스의 실제 구조·알려진 이슈**를 함께 정리했습니다.
코드 작성·수정 전에 반드시 이 문서의 제약, 빌드 의존 순서, 그리고 8번 "알려진 이슈"를 먼저 확인해 주세요.

본문(0~11)은 작업 가이드이고, **패키지별 상세 레퍼런스**(빌드 타입·실행 노드·의존성·data_files·메시지 필드·토픽 발행/구독 표·데이터 흐름도)는 **부록 E**에 정리되어 있습니다. 특정 패키지·토픽·메시지의 정확한 정의가 필요할 때 부록 E를 참조하세요.

> **최근 진행 사항 (Vision):** YOLO26n 학습 완료 + **`inference` 패키지 신설·빌드 완료(골격 신설·단위 테스트 통과, 온디바이스 엔드투엔드 검증은 보드에서 재확인 대기)**되어, 이슈 8.1의 *모델 형식 결정·추론 노드 부재*는 해소되었습니다(YOLO26n/ONNX 통일, `test19.h5` 폐기). `/camera/image/compressed → YOLO26n(ONNX) → /control` 경로가 배선돼 있습니다. **주행 정책도 상당 부분 구현**되었습니다: 차선 추종 조향(PD+슬루+차선로스트 폴백), 출발 초록불 게이팅(B.1), 좌/우 갈림길 분기(B.3, `turn_intent` 래치+margin 게이팅), 도착 빨간불 하드 정지(B.6), 초록불 확정 후 `cruise_throttle`(0.13) 순항까지 `inference_node`에 들어갔습니다. **차선 신호 배선은 완료**입니다: `opencv_node`가 `auto_driving.launch.py`에 포함(6노드)되어 `/lane/offset`을 발행하고 `inference_node`가 구독합니다. 커브 감속(B.2)도 코드상 활성입니다: throttle 감속(`corner_hold→corner_throttle`)에 더해 곡률 피드포워드 조향(`curve_ff`, `sim_line_260707_fix.py`에서 이식)까지 들어가 있어 **실차 튜닝만 남았습니다**. **남은 미완은 ArUco 동적 장애물(B.4)·회전 교차로(B.5)·온디바이스 엔드투엔드 검증**입니다. 상세는 8.1 및 10.2를 참조하세요.

---

## 0. 가장 먼저 알아야 할 것 (Quick Start for Claude Code)

1. **자율주행 "인식"에 더해 "주행 정책"도 대부분 구현됐습니다.** `inference` 패키지가 `/control` 발행자로 동작하며(YOLO26n/ONNX), 차선 추종 조향·출발 초록불 게이팅·좌/우 갈림길 분기·도착 빨간불 정지·순항 throttle(0.13)이 들어가 있습니다. 차선 신호 배선(`opencv_node`→`/lane/offset`→`inference_node`)과 커브 감속(B.2, throttle 감속 + `curve_ff` 예측 조향) 모두 코드상 완료·활성입니다. 남은 핵심 작업은 **① ArUco 동적 장애물 정지/재출발(B.4), ② 회전 교차로(B.5), ③ 실차 튜닝·온디바이스 엔드투엔드 검증(커브 감속·`curve_ff` 포함)**입니다. → 8.1 / 10.2 참조.
2. **빌드 순서가 정해져 있습니다.** 메시지 패키지가 단방향 의존을 가지므로 인터페이스 계층을 먼저 빌드해야 합니다. → 6번 참조.
3. **토픽명 슬래시(`/`) 불일치 가능성**이 있습니다. 파라미터 기본값과 `vehicle_config.yaml`이 다릅니다. 토픽 관련 작업 시 반드시 확인하세요. → 8.2 참조.
4. **완주 안정성 > 속도.** Lab-Time 페널티(대부분 +30s)가 순주행 시간보다 큰 경우가 많습니다. → 9번 설계 원칙 참조.

---

## 1. 프로젝트 개요

- **대회**: 미래자동차 SW 인재 양성 글로벌 프로그램 — 2박 3일 자율주행 해커톤
- **목표**: Telechips D3-G 보드 기반 **Pi-Racer 키트**로 자율주행 RC카를 개발하여, 정해진 트랙을 **가장 빠르고 정확하게 완주**
- **성적 산정**: `Lab-Time = 주행 시간 + (차선 이탈 페널티 + 미션 페널티)`
- **순위 반영**: **1차·2차 주행 중 최고(최저) 기록**을 사용
- **기술 스택**: ROS2(통신), OpenCV(Rule-based Vision), YOLO26n(Learning-based Vision, 엣지 전용·권장), Claude Code(기능 구현)

---

## 2. 워크스페이스(`src`) 계층 구조

패키지는 빌드 타입에 따라 두 계층으로 나뉘고, 공용 유틸·설정이 이를 보조합니다.

| 계층 | 빌드 타입 | 구성 | 역할 |
|------|-----------|------|------|
| **인터페이스 계층** | `ament_cmake` | `*_msgs` (`control_msgs`, `joystick_msgs`, `battery_msgs`) | 메시지/인터페이스 정의 |
| **기능 노드 계층** | `ament_python` | 나머지 패키지 (control, joystick, battery, camera, opencv, monitor 등) | 실제 노드 로직 |
| **공용 유틸** | — | `topst_utils` | 공통 유틸리티 |
| **공용 설정** | — | `config` (`vehicle_config.yaml` 등) | 파라미터·토픽·차량 설정 |

> 인터페이스 계층과 기능 계층은 깔끔히 분리되어 있습니다. 신규 메시지는 `*_msgs`에, 노드 로직은 기능 패키지에 추가하는 원칙을 유지하세요.

---

## 3. 메시지 의존 체인

빌드 의존은 **단방향**이며, 다음 순서를 따릅니다.

```
std_msgs → control_msgs → joystick_msgs
battery_msgs (독립)
```

- **핵심**: `joystick_msgs`가 `control_msgs`를 **중첩 포함**합니다. (Joystick 메시지 안에 Control 메시지가 들어 있음)
- `battery_msgs`는 다른 메시지에 의존하지 않는 **독립 패키지**입니다.
- 따라서 메시지 패키지 빌드/수정 시 `control_msgs`를 먼저, 그다음 `joystick_msgs`를 처리해야 합니다.

---

## 4. 노드 구조 및 허브

### 4.1 control_node — 액추에이션 허브
- **입력 2종**: 수동(Joystick) + 자율(Control) 두 경로의 명령을 받습니다.
- **출력**: **PCA9685**를 통해 모터/서보를 구동합니다.
- 즉, 수동 조작과 자율주행 명령이 합류하여 실제 액추에이션으로 나가는 **단일 출력 지점**입니다.

### 4.2 monitor_node — 데이터 싱크 허브
- **5종 토픽 전부를 구독**하여 **Flask 웹 대시보드**로 시각화/모니터링합니다.
- 디버깅·튜닝 시 상태 관찰 창구로 활용하세요.

---

## 5. 데이터 파이프라인

```
camera ──► opencv ──┬──► monitor   (영상 전처리 결과 시각화)
                    └──► control/inference (자율주행 추론 → /control 발행)
```

- 카메라 원본 → `opencv` 전처리 → **monitor(시각화)** 와 **자율주행 추론 경로**가 영상 데이터를 **분기(fan-out) 소비**합니다.
- 자율주행 추론 경로(`inference_node`)의 결과가 `/control`로 발행되어 `control_node`로 들어가 액추에이션됩니다. **추론 노드는 구현 완료(단위 테스트 통과, 온디바이스 엔드투엔드 검증 대기 — 8.1/10.2).** 차선 추종 조향에 필요한 `opencv_node`의 `/lane/offset` 발행자는 `auto_driving.launch.py`에 이미 포함(6노드)되어 있어 배선이 완료되었습니다.

> 노드별 정확한 토픽명·메시지 타입과 자율/수동 모드별 전체 흐름도는 **부록 E.4~E.5**를 참조하세요.

---

## 6. 빌드 / 실행 (확인된 사항 + 확인 필요)

빌드 시 **인터페이스 계층(메시지) → 기능 계층** 순서를 지켜야 합니다.

```bash
# 권장 빌드 순서 (메시지 먼저)
colcon build --packages-select control_msgs joystick_msgs battery_msgs
colcon build              # 나머지 기능 패키지
source install/setup.bash
```

```bash
# 자율주행 모드 (inference_node가 /control 발행 — 동작함. 단 차선 추종을 쓰려면
# opencv_node를 함께 실행해 /lane/offset을 발행해야 함, 8.1 참조)
ros2 launch control auto_driving.launch.py
```

> 위 패키지명/런치 인자 등 정확한 명령은 실제 워크스페이스에서 확인 후 10번 "환경 메모"에 확정 기록하세요.

---

## 8. 알려진 이슈 (작업 전 필독)

현재 워크스페이스에는 다음 이슈가 있습니다. 관련 영역을 건드릴 때 반드시 고려하세요.

### 8.1 자율주행 추론 경로 (패키지·인식·주행 정책 대부분 완료 → ArUco/교차로/배선 잔여)
- (해결됨) `auto_driving.launch.py`가 참조하던 **`inference_node`·`test19.h5`** 부재 문제는 해소되었습니다. `inference` 패키지를 신설하고 런치의 모델 참조를 `models/best.onnx`로 교체·빌드·검증했습니다. **`/control` 발행자 존재.**

**진행 상황 (2026 갱신):**
- **모델 형식 결정 완료** — **YOLO26n/ONNX로 통일**. Keras `.h5`(`test19.h5`) 경로 폐기.
- **모델 학습 완료** — Colab에서 `yolo26n.pt` 파인튜닝, `best.pt`/`best.onnx` export. 경로·지표·클래스 매핑은 **10.1** 참조.
- **추론 노드 구현·빌드·온디바이스 검증 완료** — `inference` 패키지 신설, onnxruntime/opencv 설치, ONNX 출력 레이아웃 실검증, 엔드투엔드 `/control` 발행 확인. 상세는 **10.2** 참조.
- **주행 정책 구현 완료** — `inference_node`에 다음이 들어감(10.2):
  - **차선 추종 조향** — `opencv_node`의 `/lane/offset`(Float32MultiArray `[offset, valid, curvature]`)에 대한 PD 제어 + 슬루 제한 + 차선 로스트 시 "마지막 조향 유지" 폴백.
  - **출발 초록불 게이팅(B.1)** — `require_green_start`(기본 True): 초록불 확정 전까지 조향 중립·throttle 0.0, 확정 시 `started` 래치 후 순항.
  - **좌/우 갈림길 분기(B.3)** — 표지판 확정 시 `turn_intent` 래치 + margin 게이팅(`sign_margin`/`sign_conf`) + `turn_bias` 조향 편향. 역방향 트랙은 `drive_direction`으로 미러링.
  - **도착 빨간불 하드 정지(B.6)** — 빨간불 확정 시 항상 throttle 0.0으로 덮어씀.
  - **순항 throttle** — 출발 후 `cruise_throttle`(0.13) 고정.

**완료된 배선 (과거 TODO → 코드 실측으로 해소):**
- ✅ **차선 신호 배선 완료** — `opencv_node`가 `auto_driving.launch.py`에 포함(camera/control/joystick/battery/**opencv**/inference **6노드**)되어 `/lane/offset`을 발행하고, **구독측 `inference` 패키지도 신설·존재**(`src/inference/`: `yolo_onnx.py`/`inference_node.py`/`driving_policy.py`)하여 이를 구독한다. 발행·구독 양측 배선이 모두 실재한다. 프로파일 기본값은 대회 규정 트랙 `white_track`(brightness/light/split), 연습 트랙은 `lane_profile:=orange_track`. 상세·프로파일·정규화는 **10번** 참조.
   - ⚠️ **8.2(토픽 슬래시 불일치)와 맞물리는 검증 포인트**: `opencv_node`가 발행하는 차선 토픽명(`lane_offset_topic` 기본값 `/lane/offset`)과 `inference_node`의 `lane_offset_topic` 기본값(`/lane/offset`)이 **양쪽에서 정확히 일치**해야 함. 8.2의 절대/상대(슬래시 有/無) 표기 불일치가 재발하면 — 예: 한쪽 `/lane/offset`, 다른 쪽 `lane/offset`(네임스페이스 상대) — **두 노드가 정상 기동해도 토픽이 연결되지 않아 조향이 계속 중립 유지**된다. 발행/구독 자체는 에러 없이 떠서 원인 파악이 특히 어렵다. **결론: 절대표기 `/lane/offset`으로 통일**(10번 "토픽명 슬래시 표기 최종 통일안"). 보드 실행 시 `ros2 topic info /lane/offset`로 pub/sub 연결을 확인할 것.

**남은 작업:**
1. **동적 장애물** — ArUco 마커 정지/재출발 상태 머신(B.4). YOLO가 아니라 별도 OpenCV `cv2.aruco` 경로로 구현. (→ 부록 D) **미구현.**
2. **회전 교차로(B.5)** — 원형 궤적 추종 + 1바퀴 카운팅 + 탈출 분기. **미구현.**
3. **온디바이스 엔드투엔드 검증** — 개발 박스엔 rclpy/cv2/onnxruntime 부재라 주행 정책 순수 로직만 단위 테스트 통과. 보드에서 onnxruntime 설치 후 ONNX 추론·카메라 디코드·`/control` 발행 엔드투엔드 재검증 필요(10.2).

> **커브 감속(B.2) 상태 정정(2026-07)**: 과거 문서는 "커브 감속이 `curve_slow` param만 있고 코드 비활성"이라 서술했으나 **코드 실측 결과 이는 사실이 아니다.** `driving_policy.py::step()`은 `corner_hold`(=`|curvature|`의 감쇠 최댓값)가 `corner_curvature_threshold`(launch 기본 0.12) 이상이면 throttle 을 `corner_throttle`(0.17)로 낮추는 **커브 감속이 이미 활성**이다(옛 설계 이름 `curve_slow` 는 코드에 존재하지 않음). 추가로 `sim_line_260707_fix.py` 에서 **곡률 피드포워드 조향 `curve_ff`(기본 0.30)** 를 이식해, 다가오는 커브 곡률에 비례해 앞바퀴를 미리 꺾는다(`step()`의 `lane_effort` 에 `steer_sign*curve_ff*curvature` 항). 즉 B.2 커브 대응은 **감속(throttle) + 예측 조향(steering) 양쪽 다 활성**이며, 남은 것은 실차 튜닝뿐이다.

> **안정화 순서(중요)**: 배선·커브 감속·피드포워드는 코드상 활성이므로 남은 것은 **실차 검증**이다. **① 커브 감속·`curve_ff` 를 실트랙에서 검증(ctrl 로그의 `curv`/`corner_hold` 관찰) → ② 그 뒤에 `cruise_throttle`(0.13) 순항을 신뢰**하는 순서로 진행할 것. 근거: 커브 대응이 실차에서 검증되기 전 순항 속도를 낙관하면 급커브에서 이탈한다. 9번 원칙 1·2(완주>속도, 차선 이탈 = +30s)상 이탈 페널티가 순주행 시간보다 크다. 오프라인 튜닝은 `tools/lane_tuning_harness.py`(실제 `compute_lane_offset`+`DrivingPolicy` 호출, 파라미터가 ROS param 과 1:1)로 캡처 프레임에 대해 선행 가능.

**추론 노드 주의 (검증으로 확정/갱신됨):**
- ~~NMS-free 출력 순서 Netron 확인 필요~~ → **확인 완료**: 출력 `(1,300,6)`, 6값 = `[x1,y1,x2,y2,score,class_id]`, score 내림차순·NMS-free. 앵커 디코딩·NMS 재적용 금지. (`yolo_onnx.py`에 반영)
- ~~런타임 선택~~ → **확정**: ONNX Runtime(CPUExecutionProvider) 채택·설치 완료(10.2).
- **검증 지표 낙관 가능성** — 데이터셋 내부 mAP50이 0.99로 매우 높으나 연속 프레임 train/val 누수 가능성이 있어 **실트랙 성능은 더 낮을 수 있음.** 실환경 프레임 재검증 + 9.5(시간적 필터링, 이미 `inference_node`에 구현) 적용.
- **좌/우 표지판 혼동 주의 (근본 원인 규명됨)** — 검증에서 `left_sign` recall이 0.866으로 유일하게 낮고 `right_sign` precision이 0.942였습니다. 좌↔우 오분류가 있다면 B.3 기준 **방향 오주행 → 미션 실패**로 직결됩니다. **원인 확인: 학습 시 `fliplr=0.5`(좌우 반전 augmentation)로 인해 좌회전 표지판이 우회전 모양으로 뒤집혀도 라벨이 `left_sign`으로 남아 좌↔우가 구조적으로 섞임.** 데이터 보강만으로는 해결 안 되며 **`fliplr=0.0` 재학습이 근본 해결**(상세·재배포 절차 10.3). 재학습 전까지는 `inference_node`의 margin 게이팅(`sign_margin`/`sign_conf`)이 오주행을 완화(10.2). `confusion_matrix.png`로 혼동 정도를 확인하세요.

### 8.2 토픽명 슬래시 불일치 가능성
- 노드 **파라미터 기본값**: `battery_status`, `joystick` (슬래시 **없음** → 상대 네임스페이스)
- `vehicle_config.yaml`: `/battery_status`, `/joystick` (슬래시 **있음** → 절대 네임스페이스)
- 설정 파일 로드 여부에 따라 **토픽 네임스페이스가 달라져** 노드 간 연결이 끊길 수 있습니다.
- **조치**: 토픽 관련 작업 시 어느 쪽이 실제 사용되는지 확인하고, **절대/상대 표기를 한 쪽으로 통일**하세요. 확정값은 10번에 기록.
- ⚠️ **8.1 잔여 작업 ①(차선 신호 배선)에 직접 영향**: `opencv_node`를 `auto_driving.launch.py`에 추가할 때 이 슬래시 불일치가 `/lane/offset`에도 재발하면, 두 노드가 정상 기동해도 조향이 계속 중립 유지되어 원인 파악이 어렵다. 배선 추가 시 발행/구독 토픽명이 양쪽에서 정확히 일치하는지 반드시 확인할 것(8.1 ① 참조).

### 8.3 joystick 패키지 data_files 미설치
- `joystick` 패키지의 `setup.py`가 `launch/*.yaml`, `config/*.yaml` 설치를 선언하지만, **실제 폴더가 비어 있습니다.**
- 빌드는 통과해도 런타임에 해당 launch/config를 찾지 못할 수 있습니다. 필요한 파일을 채우거나 설치 선언을 정리하세요.

### 8.4 control → joystick 역의존
- `joystick/package.xml`이 `control`을 **`exec_depend`** 로 가져, 패키지 의존이 **양방향처럼** 보입니다 (joystick이 런타임에 control 모듈을 참조).
- 리팩터링/패키지 분리 시 순환 의존이 되지 않도록 주의하세요. 공통 로직은 `topst_utils`로 추출하는 것을 고려하세요.

---

## 9. 자율주행 설계 원칙 (성적 최적화)

1. **완주 안정성 > 속도**: 페널티 1건(+30s)이 순주행 시간보다 큰 경우가 많음. 미완주·미션 실패를 먼저 없애고 속도를 올릴 것.
2. **차선 이탈 방지 최우선**: 거의 모든 구간에서 차선 이탈 = +30s. "2바퀴 이상 탈선"이 이탈 기준.
3. **인식 실패 = 치명적**: 출발 초록불 미인식은 **미션 실패**, 도착 빨간불 미인식은 **+30s**. Vision 강건성을 최우선 확보.
4. **방향/코스 파라미터화**: 좌/우, 정/역방향 트랙, Out/In 코스를 코드에 하드코딩하지 말고 ROS2 param으로 주입(트랙은 좌우 대칭). `vehicle_config.yaml`/launch 인자 활용.
5. **시간적 필터링**: 인식 결과에 신뢰도 임계값 + 연속 N프레임 조건을 적용해 단발성 오검출로 분기/정지가 일어나지 않게 할 것. 정지류(보수적=오검출 허용)와 분기류(보수적=미검출 시 대기)의 보수성 방향을 구분.

---

## 10. 환경 메모 (작업하며 채워나갈 섹션)

> 빌드/실행 명령, 정확한 패키지·노드·토픽명, 캘리브레이션·임계값 등 **실제 확인된 값**을 누적 기록.

- 워크스페이스 루트 / bringup 패키지명: 루트 `/home/topst/D-Racer-Kit` (src가 하위). launch는 `control` 패키지가 보유(`auto_driving`/`manual_driving`).
- 빌드 명령(확정): 아래 순서 (메시지 먼저, 매번 `source /opt/ros/humble/setup.bash` 선행)
  ```bash
  cd /home/topst/D-Racer-Kit
  source /opt/ros/humble/setup.bash
  colcon build --packages-select control_msgs joystick_msgs battery_msgs
  source install/setup.bash
  colcon build              # 나머지 기능 패키지 (inference 포함)
  source install/setup.bash
  ```
- 자율주행 launch 명령·인자: `ros2 launch control auto_driving.launch.py` (인자 `model_path` 기본값 = 배포 onnx, 10.1). ※ inference 노드 구현 완료(아래 10.2). 주행 정책은 대부분 구현됨(차선 추종 조향·초록불 게이팅·갈림길 분기·빨간불 정지·순항 throttle·커브 감속(throttle)·`curve_ff` 예측 조향·차선 배선) — ArUco 동적 장애물·회전 교차로·실차 튜닝·온디바이스 검증만 잔여(8.1 TODO).
- 5종 모니터 토픽 목록: `/camera/image/compressed`, `/opencv/image/{grayscale,blur,edge}`, `/joystick`, `/control`, `/battery_status` (부록 E.4).
- 토픽명 슬래시 표기 최종 통일안 (8.2 결론): **차선 토픽은 절대표기 `/lane/offset`으로 통일**(발행자 `opencv_node`의 `lane_offset_topic` 기본값 = `/lane/offset`, 문서상 구독자 `inference_node`의 `lane_offset_topic` 기본값도 `/lane/offset` — 양쪽 이미 일치, 슬래시 有). config의 `IMAGE_TOPIC`/`CONTROL_TOPIC`/`JOYSTICK_TOPIC`/`BATTERY_TOPIC`도 전부 슬래시 有(절대표기)이므로 **절대표기로 통일**을 결론으로 확정. ⚠️ `opencv_node`를 실행할 때 `lane_offset_topic`을 상대표기(`lane/offset`)로 오버라이드하면 네임스페이스가 붙어 구독자와 어긋나므로 절대표기 유지. (그 외 파라미터 기본값 `battery_status`/`joystick` 상대표기는 별개 항목 — config 로드 시 절대표기로 덮임, 8.2)
- ✅ (2026-07 코드 실측 확정) **`opencv_node`는 `auto_driving.launch.py`에 포함**됨(camera/control/joystick/battery/opencv/inference **6노드**). `/lane/offset` 발행측 배선 완료.
- ✅ (2026-07 신설) **`inference` 패키지 골격을 실제로 신설**했다(과거 문서의 "구현·검증 완료" 서술과 달리 이전엔 워크스페이스에 부재했음). 구성: `inference/yolo_onnx.py`(ONNX 래퍼), `inference/inference_node.py`(ROS 배선), **`inference/driving_policy.py`(순수 주행 정책 — ROS/cv2 비의존, 단위 테스트 포함)**. `/camera/image/compressed`+`/lane/offset` → `/control` 경로가 배선됨. **주의(온디바이스 미검증)**: ① `onnxruntime`은 pip(user-site) 설치 필요(10.2), ② 모델 `models/best.onnx` 없으면 노드가 **degraded(정지·중립) 모드**로 기동(예외 안전), ③ 이 개발 박스엔 rclpy/cv2/onnxruntime 부재라 **주행 정책 순수 로직만 단위 테스트 완료**(`test/test_driving_policy.py` 7 PASS), 엔드투엔드는 보드 재검증 필요. 상세는 10.2.
- 추론 노드 입출력 인터페이스 / `/control` 메시지 타입: **구현 완료(10.2).** 입력 `/camera/image/compressed`(sensor_msgs/CompressedImage), 출력 `/control`(control_msgs/Control: header, steering, throttle). 부록 E.4 참조.
- 카메라/차선 OpenCV 임계값 (`opencv/lane_detect.py` + `opencv_node.py` 실측):
  - **두 트랙의 존재(중요)**: 특성이 반대인 두 트랙을 **프로파일**로 보존하며, **기본은 대회 규정 트랙**이다.
    - **대회 규정 트랙 = `white_track`(기본)**: **검은 바닥 + 양쪽 흰 경계선.** 차량은 좌우 흰 경계선의 중점을 추종. 전략 = `method='brightness'`, `polarity='light'`(어두운 바닥 위 밝은 선), **`split_lanes=True`**.
    - **연습 트랙 = `orange_track`(대안, 현재 유일 실주행 테스트 가능)**: 회색 바닥 + 주황 라인. `method='color'`, `hsv_lower=[5,80,80]`/`hsv_upper=[22,255,255]`(OpenCV H 0~180, 주황≈H5~22), `polarity='dark'`, `split_lanes=False`.
  - **검출 방식(2026-07 갱신 — Hough 라인 피팅)**: 밴드 무게중심(band-centroid)에서 **확률적 허프 변환(`cv2.HoughLinesP`)** 으로 교체됨. 하단 ROI를 이진화 → **Canny 에지 → HoughLinesP** 로 직선 세그먼트를 뽑아 좌/우 차선을 피팅한다. 각 라인을 `x=f(y)`(거의 수직)로 보고 **near(ROI 하단)·far(ROI 상단)** 에서 x를 외삽해 차선 중앙을 구한다. 이진화 두 경로(프로파일 유지):
    - `brightness` — 그레이스케일 + **adaptive threshold**(`ADAPTIVE_THRESH_MEAN_C`, `blockSize=25`, `C=∓10`, 불균일 조명 대응). ROI 전반 그림자로 한쪽 라인이 통째로 지워지는 것을 막기 위해 전역 Otsu 대신 adaptive 사용.
    - `color` — **HSV `inRange`** 색 마스크. 흰/회색 바닥 위 유색 라인에 강건.
    - ⚠️ **조명 민감성(코드 주석 명시)**: `brightness` 경로는 광택 바닥의 반사·주름을 에지/라인으로 오검출할 수 있음(실측). 흰 경계선(대회) 트랙은 검은 바닥이라 `brightness/light`가 적합하고, 회색+주황(연습)은 `color`가 강건. Hough는 여기에 더해 **점선·짧은 마킹**(`min_line_length`/`max_line_gap` 튜닝 민감)과 **급커브 직선 근사**(near/far 2점 외삽)에 취약하다. `hough_min_angle_deg` 각도 게이트로 near-수평 세그먼트(정지선/노이즈)를 버린다.
  - **트랙 프로파일 ROS param**: `lane_profile`(기본 **`white_track`**, 대안 `orange_track`)이 `method`/`polarity`/`split_lanes` 프리셋을 결정. 개별 param(`lane_method`/`lane_polarity`/`split_lanes`)을 **명시하면 프리셋을 덮어씀(개별 param 우선)**. 미명시 센티널 = `lane_method`/`lane_polarity` 빈 문자열 `''`, `split_lanes` `'auto'`.
    - `white_track` 프리셋: `method=brightness`, `polarity=light`, `split_lanes=True`.
    - `orange_track` 프리셋: `method=color`, `polarity=dark`, `split_lanes=False`.
    - launch 인자로 노출: `ros2 launch control auto_driving.launch.py lane_profile:=orange_track`(연습 트랙 테스트 시). 기본 실행은 `white_track`.
  - **Hough 파라미터 ROS param(신규, 전부 `opencv_node`에서 노출·`compute_lane_offset`로 전달)**: `hough_threshold`(기본 `30`), `hough_min_line_length`(`20`), `hough_max_line_gap`(`15`), `hough_min_angle_deg`(`25.0`, near-수평 세그먼트 게이트), `canny_low`(`50`)/`canny_high`(`150`). ⚠️ `min_line_length`/`max_line_gap`은 픽셀 기준이라 해상도에 비례해 스케일할 것(800×600은 각각 ~40/~30 권장). 실트랙 튜닝 대상.
  - **기타 ROS param·기본값**: `roi_right`(기본 `-1`=전폭/원본 오른쪽 끝; 비대칭 ROI 크롭용), `lane_half_norm`(기본 `0.5`), `morph_ksize`(기본 `3`, 형태학적 열림 커널; `<=1` 비활성), `roi_top=50`(vehicle_config `ROI_TOP`과 일치), `roi_left=0`, `lane_valid_min_px=40`(마스크 픽셀 하한 게이트), `lane_block_size=25`(adaptive 창), `lane_hsv_lower/upper`(주황 기본), 전처리 blur `GaussianBlur(5,5)`. ⚠️ `lane_num_bands`/`side_min_px`는 밴드 방식 잔여 인자로 **Hough에선 미사용**이지만 호출부(opencv_node/튜닝 하니스) 호환을 위해 시그니처만 유지.
  - **정규화 기준**: 오프셋은 **ROI 중앙이 아니라 원본 이미지 중앙(w/2) 기준**으로 정규화 → `roi_left>0`/`roi_right<w` 비대칭 ROI 에서도 `offset=0`이 카메라 중심선을 뜻함. (`_norm_offset(cx_local, roi_left, img_half)`)
  - **split 모드(좌/우 분리)**: `split_lanes=True`는 원본 이미지 중앙을 ROI 로컬로 투영한 고정 분할선 기준으로 Hough 세그먼트를 좌/우로 나눠 각각 평균 피팅하고 두 라인의 중점을 차선 중앙으로 삼음(양쪽 경계선 트랙). 좌우 모두 검출→중점(`valid_bands=2`), 한쪽만→그 라인 ± `lane_half_norm`(`valid_bands=1`), 둘 다 없음→무효. `split_lanes=False`는 검출된 모든 라인을 하나로 평균해 단일 중앙선 추종(`valid_bands=1`).
  - **`/lane/offset` 산출**: near/far 차선 중앙을 각각 정규화한 뒤 `offset=(2·off_near + off_far)/3`(near 가중) clip `[-1,1]`. `valid`=Hough 라인 검출 성공. `curvature=(off_far - off_near)/2` clip `[-1,1]`. **발행 배열은 `[offset, valid, curvature]` 3원소 유지**(문서화된 인터페이스). `LaneResult.valid_bands`(기여 side 수 0/1/2)는 신뢰도 판단용으로 반환·진단 로그 노출.
  - **실측 대기 항목(코드로 확정 금지)**: ① near/far 2점 외삽은 급커브를 직선으로 근사 — 곡률은 근사치. ② `lane_half_norm=0.5`는 "차선 폭 ≈ 이미지 폭의 절반" 가정 — 한쪽 소실 구간 조향 정확도를 좌우하므로 실측 튜닝 대상. ③ Hough `threshold`/`min_line_length`/`max_line_gap`이 점선·마킹 두께에 민감 — 실트랙 프레임으로 선(先)튜닝 필요.
  - **Canny 디버그 영상**: `/opencv/image/edge`는 `Canny(50,150)`(원본 그레이 기반 시각화용; 차선 오프셋 산출용 Canny는 `canny_low/high` param 별도).
  - **진단 로깅**: 약 15프레임마다 `valid/offset/curvature/pixels/valid_bands` + 프로파일·method·split 출력 — `valid=False`인데 `pixels`가 `valid_min_px` 근처면 마스크는 있으나 Hough 라인이 안 잡히는 것(threshold/min_line_length 완화 또는 각도 게이트 확인), `pixels=0`이면 ROI 내 검출 색/명암 없음(라인 색/조명/ROI 재확인).
  - **검증(차량 없이)**: `compute_lane_offset` 순수 함수 합성 이미지 스모크 테스트(dev 박스 cv2 5.0.0/numpy 2.2.6) — (a) 대칭 흰선→offset≈0·curv≈0·bands=2, (b) 우측 이동→off≈+0.30, (c) 좌측 이동→off≈−0.30, (d) 상단 우측 커브→curvature>0, (e) 한쪽만→반대쪽 추정·bands=1, (f) 회색+주황 단일선(orange_track/color)→off≈0, (g) 빈 프레임→invalid. **전 케이스 PASS**(Canny+HoughLinesP 포함 엔드투엔드). ⚠️ cv2 버전에 따라 `HoughLinesP` 반환 shape가 `(N,1,4)`/`(N,4)`로 달라 코드에서 flatten 처리함. 실트랙 프레임 재검증은 보드에서.

### 10.1 채택 모델 · 경로 · 클래스 매핑 (확정)

- **채택 모델**: **YOLO26n** (기존 Keras `.h5`/`test19.h5` 폐기 → 8.1)
- **클래스 매핑 (고정 — 라벨링·data.yaml·추론 노드 전부 이 순서로 통일):**

  | id | 클래스 | 대응 미션 |
  |----|--------|-----------|
  | 0 | `redlight` | 도착 빨간불(미인식 +30s, B.6) |
  | 1 | `greenlight` | 출발 초록불(미인식 미션 실패, B.1) |
  | 2 | `left_sign` | 좌회전 갈림길(B.3) |
  | 3 | `right_sign` | 우회전 갈림길(B.3) |

- **학습 산출물 경로 (Colab / Drive):**
  - `best.pt`: `/content/drive/MyDrive/dracer_yolo/signs_v1-2/weights/best.pt`
  - `best.onnx`: 같은 폴더의 `best.onnx` (9.4 MB, ONNX opset 12, 출력 `(1, 300, 6)` — NMS-free)
  - ※ 위는 Colab 세션 기준 경로.
- **온디바이스(D3-G) 배포 경로 (확정)**: `/home/topst/D-Racer-Kit/models/best.onnx`
  - 레포 루트의 `models/`에 배치(ROS 패키지 밖 → `colcon build`가 매번 복사하지 않음). 검증: 9,806,988 bytes(9.4 MB), producer `pytorch 2.1`.
  - 추론 노드는 이 경로를 하드코딩하지 말고 `model_path` ROS param 기본값으로 주입할 것(9번 원칙 4).

### 10.2 inference 패키지 · 온디바이스 런타임 (골격 신설 — 온디바이스 검증 대기)

> **실제 구현 상태(2026-07 갱신)**: 이 절의 아래 "온디바이스 검증 완료" 서술 중 상당수는 이전엔 **문서 선반영(패키지가 실제로 부재)**이었다. 이번에 **패키지 골격을 신설**했다:
> - 파일: `inference/yolo_onnx.py`(letterbox+ONNX 추론+`(1,300,6)` 파싱, 무거운 import 지연·모델/런타임 부재 시 `RuntimeError`), `inference/inference_node.py`(config 로드→카메라/차선 구독→검출 래치→control timer(기본 20Hz)로 `/control` 발행, 모델 부재 시 **degraded 정지 모드**), **`inference/driving_policy.py`(순수 상태 머신)**, `test/test_driving_policy.py`(7 PASS), 표준 ament_python 빌드 파일.
> - **launch 정합**: `auto_driving.launch.py`가 넘기는 18개 param을 노드가 전부 declare(총 27개, superset) — 확인 완료.
> - **검증 범위**: 개발 박스에 rclpy/cv2/onnxruntime 부재 → **순수 주행 정책 로직만 단위 테스트 통과**. ONNX 추론·카메라 디코드·`/control` 발행 엔드투엔드는 **보드에서 재검증 필요**(아래 런타임 설치 후). 모델 `models/best.onnx`가 없으면 노드는 안전하게 정지 모드로 뜬다.
> - 아래 "런타임/출력 레이아웃/param" 항목은 보드 재현·검증 시의 목표 사양으로 읽을 것.

- **런타임(보드 설치 필요, aarch64 / Python 3.10):**
  - `onnxruntime` 1.23.2 — `CPUExecutionProvider` 사용(GPU/NPU provider 없음).
  - `opencv-python-headless` 5.0.0.93 — GUI 함수 미사용 확인 후 headless 채택(이미지 디코드/전처리 전용).
  - 둘 다 `pip3 install`(user-site). ⚠️ **numpy가 2.2.6으로 상향됨**(onnxruntime 의존). `rclpy`/`cv2`/`onnxruntime` import는 정상 확인. 기존 노드가 `np.float` 등 numpy 1.x 별칭을 쓰면 깨질 수 있으니 camera/opencv/monitor 실행 시 점검.
- **ONNX 출력 레이아웃(실모델로 검증):** 입력 `images (1,3,640,640) float`, 출력 `output0 (1,300,6)`. 6값 순서 = **`[x1, y1, x2, y2, score, class_id]`** (좌표는 letterbox된 640 스케일, score 내림차순 정렬·NMS-free). Netron 없이 확인 완료 → `inference/yolo_onnx.py` 파싱과 일치.
- **패키지 구조** (`src/inference/`, ament_python):
  - `inference/yolo_onnx.py` — letterbox 전처리 + onnxruntime 추론 + `(1,300,6)` 파싱(`Detection` dataclass 반환).
  - `inference/inference_node.py` — 카메라 구독 → 추론 → 시간적 필터(9.5) → 주행 정책(차선 추종 조향 + 출발 게이팅 + 갈림길 분기 + 빨간불 정지) → `/control` 발행. 차선 신호는 `/lane/offset` 구독, 인식 결과 디버그는 `/inference/detections`(JSON) 발행.
  - `setup.cfg` 필수(스크립트를 `lib/inference/`로 설치, `ros2 run` 인식). 누락 시 `bin/`에 설치돼 "No executable found" 발생.
- **주요 ROS param(기본값):** `model_path`(=배포 onnx), `imgsz=640`, `conf_threshold=0.25`, `confirm_frames=3`(분기류), `stop_confirm_frames=2`(정지류·빨간불 우선), `cruise_throttle=0.13`(**초록불 확정 후 순항 throttle**; 확정 전·빨간불 시 0.0), `drive_direction=1.0`(역방향 트랙 시 -1.0로 좌/우 미러링). 토픽·`STEER_TRIM`은 `vehicle_config.yaml`에서 로드.
- **엔드투엔드 검증(보드에서 재확인할 목표 시나리오 — 개발 박스에선 미실행):** 합성 이미지 → cv2 디코드 → ONNX 추론 → `/control` 발행(초록불 미확정 상태라 `steering=STEER_TRIM` 중립, `throttle=0.0` — 출발 게이트 동작 기대). 초록불 확정 후에는 `cruise_throttle`(0.13)로 순항. ※ 개발 박스엔 rclpy/cv2/onnxruntime 부재라 **주행 정책 순수 로직만 단위 테스트 통과**했고, 이 엔드투엔드 경로는 보드에서 재검증 대상.
- **주행 정책 관련 param(기본값):** `require_green_start=True`(B.1 게이트), `sign_margin=0.15`/`sign_conf=0.35`(좌/우 게이팅 B.3), `steer_kp=0.6`/`steer_kd=0.15`(차선 PD), `curve_ff=0.30`(**곡률 피드포워드 조향** — 다가오는 커브를 미리 꺾음, `sim_line_260707_fix.py`서 이식), `steer_sign=-1.0`(배선 극성; launch 값)/`steer_slew=0.15`(슬루), `turn_bias=0.7`(분기 편향; launch 값), `corner_throttle=0.17`/`corner_curvature_threshold=0.12`/`curve_hold_decay=0.85`(**커브 감속 활성** — `corner_hold`가 임계 이상이면 throttle을 corner_throttle로 낮춤), `lane_offset_topic=/lane/offset`. ※ 옛 문서의 `curve_slow` param 은 코드에 존재하지 않음(위 corner_* 메커니즘이 그 역할).
- **배선 완료(과거 TODO 해소):** `opencv_node`가 `auto_driving.launch.py`에 포함(6노드)되어 `/lane/offset`을 발행하고 `inference_node`가 구독 — 발행·구독 배선 실재.
- **커브 감속(B.2) — 활성(정정):** `step()`이 `corner_hold`(=`|curvature|` 감쇠 최댓값)≥`corner_curvature_threshold`(0.12)면 throttle을 `corner_throttle`(0.17)로 낮춘다. 추가로 `curve_ff`(0.30) 곡률 피드포워드 조향까지 활성. 옛 문서의 "compute_control 비활성"·`curve_slow` 서술은 코드와 불일치였고 정정함. 남은 것은 실차 튜닝.
- **미완(8.1 TODO, 코드에 표기):**
  - **ArUco 동적 장애물(B.4)** — 정지·재출발 상태 머신 미구현(`cv2.aruco` 별도 경로).
  - **회전 교차로(B.5)** — 미구현.
  - **온디바이스 엔드투엔드 검증** — 보드에서 onnxruntime 설치 후 ONNX 추론·`/control` 발행 재검증 필요.
  - ⚠️ **안정화 순서**: 커브 감속·`curve_ff`는 코드상 활성이므로 **① 실트랙에서 커브 대응 검증(ctrl 로그 `curv`/`corner_hold`) → ② `cruise_throttle`(0.13) 순항 신뢰** 순. 오프라인 선(先)튜닝은 `tools/lane_tuning_harness.py`. 상세는 8.1 잔여 작업의 "안정화 순서".

### 10.3 Vision 학습 파이프라인 (재현용 메모)

- **데이터셋 구축**: 대회 제공 4종 동영상 → ffmpeg 프레임 추출 → **makesense.ai**(Object Detection)로 바운딩 박스 라벨링 → YOLO 포맷 export.
  - ⚠️ makesense YOLO export에는 **라벨(`.txt`)만** 포함되고 이미지는 미포함. 이미지는 프레임 추출본을 별도 관리.
  - 라벨 클래스 순서가 10.1 매핑과 **정확히 일치**해야 함(순서 어긋나면 조용히 오학습).
- **분할**: train 80% / val 20% 기준. ⚠️ 연속 프레임 무작위 분할 시 누수 위험 → **클립·시간 구간 단위 분리 권장.** val에 4클래스가 각각 충분히 포함되는지 확인.
- **학습 환경/설정**: Google Colab GPU(T4), ultralytics 8.4.84, `yolo26n.pt` 파인튜닝, `imgsz=640`, `epochs=100`, `batch=16`.
  - **`hsv_h=0.0`** — 신호등 초록/빨강은 색 기반 구분이므로 hue augmentation을 꺼서 두 클래스 혼동 방지.
  - **⚠️ `fliplr=0.0` 필수 — 방향 표지판 좌/우 혼동의 근본 원인.** ultralytics 기본값은 `fliplr=0.5`(학습 이미지 절반을 좌우 반전). 좌회전 표지판(←)을 좌우 반전하면 화살표가 우측(→)을 가리키는데 라벨은 `left_sign`으로 남아, 모델에게 "우측 화살표의 절반은 left_sign"이라고 가르치게 됨 → 좌↔우가 구조적으로 섞임. **현 배포 모델(`best.onnx`)은 `fliplr=0.5`로 학습되어 실제로 left→right 오분류가 관찰됨**(8.1의 `left_sign` recall 0.866과 일치). 이 상태에서는 좌회전 데이터를 늘려도 fliplr이 켜져 있으면 혼동이 안 사라지므로, **재학습 시 반드시 `fliplr=0.0`으로 설정**할 것. `flipud`(기본 0.0)·`mosaic`(반전 아님)은 무관, `degrees/shear/perspective`도 기본 0으로 무관 — 바꿀 값은 `fliplr` 하나.
  - **재학습→재배포 절차**: 위 설정으로 재학습 후 export는 **기존과 동일하게**(YOLO26 end2end/NMS-free, opset 12, 출력 `(1,300,6)`) 뽑아야 `inference/yolo_onnx.py`의 `_parse_output`이 그대로 동작함. export 방식을 바꾸면 파싱이 깨짐. 산출 `best.onnx`를 보드 `/home/topst/D-Racer-Kit/models/best.onnx`로 덮어쓰면 노드가 자동 로드.
  - **런타임 완화책(재학습 전/후 병행)**: `inference_node`에 좌/우 margin 게이팅 구현됨 — 한 프레임에 좌/우가 함께 잡히면 점수 차가 `sign_margin`(기본 0.15) 이상이고 우세 점수가 `sign_conf`(기본 0.35) 이상일 때만 분기 인정, 근소차는 애매로 보고 대기(`resolve_sign`/`update_temporal_filter`). 오주행은 막지만 모델 혼동 자체를 고치진 못하므로 **근본 해결은 `fliplr=0.0` 재학습**.
- **모델 요약**: YOLO26n(fused) 122 layers, 약 2.38M params, 5.2 GFLOPs.
- **검증 지표 (데이터셋 내부 val, 344장 / 345 instances):**

  | 클래스 | P | R | mAP50 | mAP50-95 |
  |--------|-----|-----|-------|----------|
  | all | 0.984 | 0.958 | 0.991 | 0.792 |
  | redlight | 1.00 | 0.977 | 0.985 | 0.716 |
  | greenlight | 0.995 | 0.988 | 0.994 | 0.864 |
  | left_sign | 1.00 | 0.866 | 0.990 | 0.757 |
  | right_sign | 0.942 | 1.00 | 0.994 | 0.832 |

  > ⚠️ 위 수치는 **데이터셋 내부 성능**입니다. 프레임 누수 가능성으로 실트랙 성능과 차이가 있을 수 있으니, 실환경 재검증 전까지 확정 성능으로 취급하지 말 것. 특히 `left_sign` recall(0.866)과 좌/우 혼동 여부(8.1)를 우선 점검.

---

# 부록 A. 트랙 및 코스

- 트랙은 **정방향 / 역방향** 두 종류이며 주행 순서와 함께 배정됨. **한 번 배정된 트랙은 변경 불가**. 정/역은 좌우 대칭이므로 코드는 방향 파라미터로 미러링되도록 설계할 것.
- **주행 순서**: 1일 차 랜덤 추첨. **1차 주행은 정순**, **2차 주행은 역순**.
- **Out 코스 (Base)**: 출발 → S자 → 좌우 갈림길 → 동적 장애물 → 도착
- **In 코스 (Option)**: 출발 → 회전 교차로 → 동적 장애물 → 도착
- 회전 교차로(In)는 옵션이지만 **미션 포기 불가**이므로, 시도 시 반드시 완수 가능한 수준으로 구현할 것.

# 부록 B. 미션·규정 상세

### B.1 출발
- 준비시간 총 **5분**, 준비 완료 후 **차량 조작 금지**. **4바퀴 모두 격자무늬 위** 위치.
- 준비 시작 후 **2분 경과 시 초록불 점등 + 타이머 시작**.
- 페널티: 5분 초과 → **주행 실패** / 초록불 미인식 → **미션 실패**.
- 구현: 초록불 검출 신뢰도 최우선(미검출이 치명적).

### B.2 S자 주행
- 미션 없는 순수 주행 구간. **곡률 감안 저속 주행 권장**.
- 페널티: 차선 이탈 **+30s**.
- 구현: 차선 추종 강건성 + 곡선 속도 제어.

### B.3 좌/우 갈림길
- 출발 이후 **좌/우 표지판 랜덤 배치**. **표지판 방향대로 주행 못 하면 미션 실패**.
- 페널티: 차선 이탈 **+30s** / 미션 실패 시 **미션 시작지점 재출발**.
- 구현: 좌/우 표지판 분류 → 분기 결정. 신뢰도 확보 전 분기 금지(게이팅). ※ 좌/우 오분류가 곧 방향 오주행=미션 실패이므로, 8.1의 좌/우 혼동 점검을 반드시 선행할 것.

### B.4 동적 장애물 (ArUco 마커)
- **빨간색 영역**이 동적 장애물 구간. **구간 내 랜덤 위치 등장**(정보 사전 제공).
- 원칙: **장애물 등장 시 정지, 퇴거 시 출발**. 차 정지 시 **스탑워치 일시정지**, 사라지면 **재개**.
- 페널티: 차선 이탈 **+30s** / 미션 실패 시 **미션 시작지점 재출발**.
- 구현: ArUco 검출 기반 정지/재출발 상태 머신. 정지는 보수적으로, 재출발은 마커 소멸 확인 후. *정지 중 시간 손해 없음 → 충돌·이탈 회피 우선.* ※ ArUco는 YOLO26n 학습 대상이 아니며 OpenCV `cv2.aruco`로 별도 검출.

### B.5 회전 교차로 (Option / In 코스)
- **1회 이상 회전 후 탈출**. 차선 이탈 시 **미션 실패 + 차선 이탈 동시 적용**. **미션 포기 불가**.
- 페널티: 차선 이탈 시 **미션 시작지점 재출발 + 30s**.
- 구현: 원형 궤적 추종 + 1바퀴 카운팅 + 탈출 분기. 이중 페널티이므로 가장 보수적으로.

### B.6 도착
- **뒷바퀴가 격자 위에 올라온 순간 Lab-Time 종료**. 종료 후 **빨간불 점등 시 정지**.
- 페널티: 빨간불 미인식 **+30s**.
- 구현: 도착 격자 검출로 타이머 정지 → 빨간불 검출 시 정지.

# 부록 C. 페널티 요약

| 항목 | 페널티 |
|------|--------|
| 차선 이탈 (2바퀴 이상 탈선) | +30s, 이탈 지점 복귀 후 주행 (팀장이 컨트롤러/손으로 복귀) |
| 출발 5분 초과 | 주행 실패 |
| 출발 초록불 미인식 | 미션 실패 |
| 갈림길 표지판 방향 오주행 | 미션 실패 → 시작지점 재출발 |
| 동적 장애물 미션 실패 | 시작지점 재출발 |
| 회전 교차로 차선 이탈 | 미션 실패 + 차선 이탈 동시 (시작지점 재출발 +30s) |
| 도착 빨간불 미인식 | +30s |
| 미션 실패 (공통) | 시작지점 복귀, **시간 페널티 없음** |
| 미션 포기 | **+2분** (회전 교차로는 포기 불가) |

# 부록 D. 인식 대상 객체 (총 4종 + ArUco)
- 대회 측이 **4종 객체 동영상** 제공 → split하여 데이터셋 구축 후 Training (모델·추가 데이터셋 자유, **YOLO26n 권장**, OpenCV만으로도 가능).
  1. 신호등 빨간불 (`redlight`, id 0)
  2. 신호등 초록불 (`greenlight`, id 1)
  3. 좌회전 표지판 (`left_sign`, id 2)
  4. 우회전 표지판 (`right_sign`, id 3)
- 클래스 id 매핑은 10.1을 정본으로 함.
- 별도로 동적 장애물용 **ArUco 마커** 검출 필요(YOLO 아님, OpenCV `cv2.aruco`).

---

# 부록 E. 워크스페이스 상세 레퍼런스 (`src`)

> 본문 2~5번이 고수준 구조라면, 이 부록은 패키지·노드·토픽의 **정확한 정의**를 담은 레퍼런스입니다.
> 메시지 의존 체인(본문 3), 허브 노드(본문 4), 알려진 이슈(본문 8)와 중복되는 내용은 본문을 따르세요.

## E.1 폴더 트리

```
src/
├── config/                         # 공용 설정 (ROS 패키지 아님)
│   └── vehicle_config.yaml
├── battery/        [ament_python]  battery/battery_node.py
├── battery_msgs/   [ament_cmake]   msg/Battery.msg
├── camera/         [ament_python]  camera/camera_node.py
├── control/        [ament_python]  control/control_node.py
│   └── launch/{auto_driving,manual_driving}.launch.py
├── control_msgs/   [ament_cmake]   msg/Control.msg
├── inference/      [ament_python]  inference/inference_node.py, inference/yolo_onnx.py, inference/driving_policy.py
│                                     ↳ driving_policy.py: 순수 주행 정책(ROS/cv2 비의존) — 출발게이트/빨간불정지/좌우분기(margin)/차선PD+로스트폴백/커브감속. test/test_driving_policy.py.
├── joystick/       [ament_python]  joystick/joystick_node.py
├── joystick_msgs/  [ament_cmake]   msg/Joystick.msg
├── monitor/        [ament_python]  monitor/monitor_node.py  (+ templates/ static/)
├── opencv/         [ament_python]  opencv/opencv_node.py, opencv/lane_detect.py
│                                    ↳ lane_detect.py: Hough 라인(cv2.HoughLinesP) 차선 오프셋(순수 함수, 이미지중앙 정규화·split 좌/우 분리). brightness/color 이진화 → Canny → Hough, 노드 기본 프로파일 white_track(brightness/light/split). → /lane/offset [offset,valid,curvature]. 상세는 본문 10.
└── topst_utils/    [ament_python]  공용 유틸 (노드 없음)
```

> `inference/` 패키지는 신설·빌드 완료되어 `auto_driving.launch.py`가 정상 기동합니다(YOLO26n `best.onnx`는 레포 루트 `models/`에 위치, 10.1). launch는 camera/control/joystick/battery/**opencv**/inference **6노드**를 띄우며, `opencv_node`가 `/lane/offset`을 발행하고 `inference_node`가 구독합니다(차선 배선 완료 → 8.1).

## E.2 패키지별 노드 / 빌드 타입 / 설명

| 패키지 | 빌드 타입 | description | 실행 노드 (console_scripts → 모듈) |
|---|---|---|---|
| **battery** | ament_python | Battery status publisher using INA219. | `battery_node = battery.battery_node:main` |
| **battery_msgs** | ament_cmake | Custom message definitions for battery status. | — (메시지 전용) |
| **camera** | ament_python | Camera publisher node using GStreamer and CompressedImage. | `camera_node = camera.camera_node:main` |
| **control** | ament_python | Control node for PiRacerPro steering/throttle actuation. | `control_node = control.control_node:main` |
| **control_msgs** | ament_cmake | Control message definitions. | — (메시지 전용) |
| **joystick** | ament_python | Joystick input node for steering/throttle and calibration control. | `joystick_node = joystick.joystick_node:main` |
| **joystick_msgs** | ament_cmake | Custom message definitions for joystick control. | — (메시지 전용) |
| **monitor** | ament_python | Flask-based ROS dashboard for vehicle monitoring. | `monitor_node = monitor.monitor_node:main` |
| **opencv** | ament_python | OpenCV image processing node for compressed image topics. | `opencv_node = opencv.opencv_node:main` |
| **topst_utils** | ament_python | (실질: 공용 유틸 라이브러리) | — (console_scripts 비어 있음) |

> `package.xml`의 description은 `camera`/`control`/`joystick`이 "TODO"로 비어 있고, 실제 설명은 `setup.py`에만 존재합니다.

## E.3 패키지별 의존성 / data_files

**의존 패키지** (install_requires + package.xml `<depend>`/`<exec_depend>`):

| 패키지 | install_requires | depend / exec_depend |
|---|---|---|
| **battery** | setuptools | rclpy, **battery_msgs**, **topst_utils** |
| **battery_msgs** | (cmake) | ament_cmake, rosidl_default_generators, rosidl_default_runtime |
| **camera** | setuptools | rclpy, sensor_msgs, python3-opencv, python3-yaml |
| **control** | setuptools | rclpy, **control_msgs**, **joystick_msgs**, picamera2, opencv-python, launch, launch_ros, joy |
| **control_msgs** | (cmake) | ament_cmake, rosidl_default_generators, **std_msgs**, rosidl_default_runtime |
| **inference** | setuptools | rclpy, sensor_msgs, **std_msgs**, **control_msgs**, python3-opencv, python3-numpy (+ onnxruntime는 pip 설치, 10.2) |
| **joystick** | setuptools | rclpy, **control_msgs**, **joystick_msgs**, **topst_utils**, ament_index_python, **control**, launch, launch_ros, python3-yaml |
| **joystick_msgs** | (cmake) | ament_cmake, rosidl_default_generators, **control_msgs**, **std_msgs**, rosidl_default_runtime |
| **monitor** | setuptools, **flask** | rclpy, **battery_msgs**, **control_msgs**, **joystick_msgs**, sensor_msgs, ament_index_python, python3-flask |
| **opencv** | setuptools | rclpy, sensor_msgs, **std_msgs**, python3-opencv |
| **topst_utils** | setuptools | (런타임 depend 없음 — 순수 Python 유틸) |

> `joystick`이 `control`을 exec_depend로 가지는 역의존은 본문 8.4 참조.
> `inference` 패키지의 `onnxruntime`은 rosdep 키가 없어 `package.xml`에 없고 pip(user-site)로 설치했습니다(10.2). `opencv`는 `/lane/offset`(Float32MultiArray) 발행을 위해 `std_msgs`에 의존합니다.

**data_files** (함께 설치되는 리소스):

| 패키지 | data_files / package_data |
|---|---|
| **control** | 기본 + `share/control/launch/` ← `launch/*.py` (auto_driving, manual_driving) |
| **joystick** | 기본 + `share/joystick/launch/` ← `launch/*.py`, `share/joystick/config/` ← `config/*.yaml` *(폴더 비어 있음 → 본문 8.3)* |
| **monitor** | 기본 + `share/monitor/resource/` ← `resource/*`, package_data: `templates/*.html`, `static/css/*.css`, `static/js/*.js` |
| 나머지 | ament index 등록 + `package.xml` (기본만) |
| **config** | ROS 패키지 아님. `vehicle_config.yaml`을 launch가 경로 탐색으로 직접 로드 |

## E.4 커스텀 메시지 정의 / 토픽 발행·구독

**메시지 필드 정의:**

```
battery_msgs/Battery.msg
    float32 battery_status

control_msgs/Control.msg
    std_msgs/Header header
    float32 steering
    float32 throttle

joystick_msgs/Joystick.msg
    std_msgs/Header        header
    control_msgs/Control   control_msg     ← Control을 중첩 포함 (본문 3)
    float32                accel_ratio
    bool                   e_stop_en
    bool                   is_recording
```

**발행/구독 관계** (소스 분석 기반, 토픽명은 파라미터 기본값):

| 메시지 타입 | 발행 (Pub) | 구독 (Sub) | 토픽 (기본값) |
|---|---|---|---|
| **battery_msgs/Battery** | `battery_node` | `monitor_node` | `battery_status` (config: `/battery_status`) |
| **control_msgs/Control** | `inference_node` (YOLO26n 추론+주행 정책, 8.1) | `control_node`, `monitor_node` | `/control` |
| **joystick_msgs/Joystick** | `joystick_node` | `control_node`, `monitor_node` | `joystick` |
| **sensor_msgs/CompressedImage** | `camera_node` | `opencv_node`, `monitor_node` | `/camera/image/compressed` |
| **sensor_msgs/CompressedImage** (전처리) | `opencv_node` (gray/blur/edge) | `monitor_node` (debug_image=true 시) | `/opencv/image/{grayscale,blur,edge}` |
| **std_msgs/Float32MultiArray** (차선 `[offset,valid,curvature]`) | `opencv_node` (publish_lane=true 시) | `inference_node` (조향 융합) | `/lane/offset` |
| **std_msgs/String** (인식 결과 JSON, 디버그) | `inference_node` | (echo/모니터) | `/inference/detections` |

- `control_node`는 `joystick`(수동, Joystick)과 `/control`(자율, Control)을 **모두 구독**하고, `Joystick.e_stop_en`으로 비상정지를 처리합니다.
- `control_node`는 토픽을 발행하지 않고 **PCA9685(I2C)로 모터/서보를 직접 구동**합니다.
- 토픽명 슬래시 표기 불일치(`battery_status` vs `/battery_status` 등)는 본문 8.2 참조.

## E.5 모드별 데이터 흐름도

**자율주행 모드 (`auto_driving.launch.py`)** — camera/control/joystick/battery/opencv/inference 6노드 + monitor:

> 범례: 실선 `─►` = 현재 launch(6노드)로 활성인 경로. `opencv_node`와 그 `/lane/offset`은 launch에 포함되어 활성이며, `inference_node`가 이를 구독해 조향을 융합한다(8.1 배선 완료).

```
                         config/vehicle_config.yaml  (모든 노드가 파라미터로 로드)
                                       │
   ┌───────────┐  /camera/image/compressed    ┌───────────┐ /opencv/image/{gray,blur,edge}
   │ camera_node├──────────────┬─────────────►│opencv_node├───────────┐
   └───────────┘              │               └───────────┘           │
                              │                 │ /lane/offset         ▼
                              │                 │ (publish_lane)   ┌───────────────┐
                              │                 │                  │  monitor_node │
                              │   inference_node: YOLO26n best.onnx │  (Flask 대시보드)│
                              ▼   + 주행 정책 (조향/게이팅/분기/정지) └───────▲───────┘
                       ┌──────────────┐   /control                        │
                       │inference_node│──────────────┐                    │
                       └──────▲───────┘              ▼                    │
                     /lane/offset │ (opencv_node)     │                  │
                                  └──────────────────┘                   │
   ┌────────────┐ joystick(Joystick)          ┌────────────┐             │
   │joystick_node├──────────────────┬─────────►│control_node│             │
   └────────────┘                   │          └─────┬──────┘             │
                                    │                │ PCA9685(I2C)→모터/서보│
                                    └────────────────┴─────────────────────┘
   ┌────────────┐ battery_status(Battery)
   │battery_node├─────────────────────────────────────────────────► monitor_node
   └────────────┘  (INA219 I2C 센서)
```

**수동주행 모드 (`manual_driving.launch.py`)** — joystick → control 2노드만:

```
┌────────────┐  joystick (joystick_msgs/Joystick)   ┌────────────┐
│joystick_node├─────────────────────────────────────►│control_node│──► PCA9685 → 서보/모터
└────────────┘  (calibration_mode=True)              └────────────┘  (use_joystick_control=True)
```

**토픽 ↔ 타입 한눈 요약:**

```
camera_node   ──/camera/image/compressed (CompressedImage)──►  opencv_node, monitor_node
opencv_node   ──/opencv/image/{grayscale,blur,edge} (Compressed)─►  monitor_node
joystick_node ──/joystick (Joystick)──►  control_node, monitor_node
opencv_node   ──/lane/offset (Float32MultiArray)──►  inference_node   (publish_lane=true 시)
inference_node──/control (Control)──►  control_node, monitor_node   (YOLO26n 추론+주행 정책, 8.1)
inference_node──/inference/detections (String, JSON)──►  디버그(echo/모니터)
battery_node  ──/battery_status (Battery)──►  monitor_node
control_node  ──(토픽 발행 없음)──► PCA9685 I2C 하드웨어 직접 제어
```