from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def get_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


def get_default_model_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'models' / 'best.onnx'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer-Kit/models/best.onnx'


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    default_model_path = get_default_model_path()
    model_path = LaunchConfiguration('model_path')
    # 라인 트래킹 점검용 런타임 인자. 평상시 기본값은 정상 주행(초록불 게이트 ON,
    # 순항 0.175). 점검 시 require_green_start:=false cruise_throttle:=0.0 로 주면
    # 바퀴는 안 굴러가고 조향만 차선에 반응한다.
    require_green_start = LaunchConfiguration('require_green_start')
    cruise_throttle = LaunchConfiguration('cruise_throttle')
    # 차선 트랙 프로파일. 기본은 대회 규정 트랙(검은 바닥+양쪽 흰 경계선):
    # brightness/light. 연습 트랙(회색 바닥+주황 라인)에서
    # 실주행 테스트할 때만 lane_profile:=orange_track 으로 전환.
    lane_profile = LaunchConfiguration('lane_profile')
    # 출발 직진 유예(제어 프레임): 초록불 출발 확정 후 이 프레임 수 동안 표지판
    # 분기를 억제하고 직진(차선 추종)한다. 실코스가 출발→S자→갈림길 순이라
    # 출발 직후 (오)검출로 즉시 꺾이는 것을 막는다. 20Hz 기준 100≈5s.
    # 실제 갈림길까지 걸리는 시간에 맞춰 튜닝(짧으면 조기 분기, 길면 분기 놓침).
    start_straight_frames = LaunchConfiguration('start_straight_frames')
    # 출발 킥스타트 지속(초): 초록불 확정 직후 조향 없이 고정 throttle 로 직진.
    # 0 으로 주면 킥 비활성(초록불 확정 즉시 정상 차선 추종 시작).
    start_kick_seconds = LaunchConfiguration('start_kick_seconds')
    # 갈림길 커밋 근접 게이팅(B.3) — 실트랙 캘리브레이션 대상이라 런타임 인자로
    # 노출한다. 재빌드 없이 sign_commit_ratio:=0.62 처럼 바로 조정.
    sign_proximity_metric = LaunchConfiguration('sign_proximity_metric')
    sign_commit_ratio = LaunchConfiguration('sign_commit_ratio')

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value=default_model_path,
            description='Path to the YOLO26n ONNX model file used by inference_node',
        ),
        DeclareLaunchArgument(
            'require_green_start',
            default_value='true',
            description='초록불 확정 전 정지 게이트(B.1). 점검 시 false 로 즉시 출발.',
        ),
        DeclareLaunchArgument(
            'cruise_throttle',
            default_value='0.16',
            description='직진 순항 throttle. 조향만 점검하려면 0.0 으로 주면 바퀴가 안 돈다.',
        ),
        DeclareLaunchArgument(
            'lane_profile',
            default_value='white_track',
            description=('차선 트랙 프로파일. 기본 white_track(대회: 검은 바닥+흰 '
                         '경계선, brightness/light). 연습 트랙은 orange_track.'),
        ),
        DeclareLaunchArgument(
            'start_straight_frames',
            default_value='100',
            description=('출발 직진 유예(제어 프레임, 20Hz 기준 100≈5s). 초록불 출발 '
                         '후 이 구간 동안 표지판 분기를 억제하고 직진한다. 0=비활성.'),
        ),
        DeclareLaunchArgument(
            'start_kick_seconds',
            default_value='0.0',
            description=('출발 킥스타트 지속(초). 초록불 확정 직후 조향 없이 고정 '
                         'throttle 로 직진 출발. 0=킥 비활성(기본). 켜려면 예: 2.0.'),
        ),
        DeclareLaunchArgument(
            'sign_proximity_metric',
            default_value='bottom_y',
            description=('갈림길 커밋 근접 지표. bottom_y(표지판이 프레임 아래로 '
                         '내려오는 정도 — 높은 카메라 권장) / area / height.'),
        ),
        DeclareLaunchArgument(
            'sign_commit_ratio',
            default_value='0.60',
            description=('근접지표 ≥ 이 값이면 갈림길 커밋(꺾기) 시작. 실트랙에서 '
                         'tools/sign_commit_calibration.py 로 캘리브레이션.'),
        ),
        Node(
            package='camera',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),
        Node(
            package='control',
            executable='control_node',
            name='control_node',
            output='screen',
            parameters=[
                {
                    'use_joystick_control': False,
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),
        Node(
            package='joystick',
            executable='joystick_node',
            name='gamepad_publisher',
            output='screen',
            parameters=[
                {
                    'calibration_mode': False,
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),
        Node(
            package='battery',
            executable='battery_node',
            name='battery_node',
            output='screen',
        ),
        Node(
            package='opencv',
            executable='opencv_node',
            name='opencv_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                    # 차선 오프셋 발행(/lane/offset) — inference_node 조향 입력.
                    'publish_lane': True,
                    # 트랙 프로파일이 검출 방식(method/polarity)을 결정. 기본
                    # white_track(대회). 개별 검출 param(lane_method/lane_polarity)을
                    # 명시하면 프리셋을 덮어쓴다(opencv_node 참조).
                    'lane_profile': lane_profile,
                    # 아래 값은 발행 해상도(vehicle_config IMAGE_WIDTH/HEIGHT,
                    # 640×480) 기준. lane_block_size·morph_ksize 는 BEV warp 전
                    # 원본 프레임에 적용되므로 해상도에 비례해 스케일했다(×2.0).
                    # lane_valid_min_px 는 고정 크기 BEV(200×240) 마스크에 대한
                    # 값이라 입력 해상도와 무관 → 유지.
                    # ⚠️ BEV 4점(bev_src_*)·슬라이딩 윈도우 param 은 opencv_node
                    # 기본값을 쓰며 실트랙 캘리브레이션 대상(CLAUDE.md 10 참조).
                    'lane_valid_min_px': 250,  # BEV 이진 마스크 픽셀 하한(미만 조기 무효)
                    'morph_ksize': 5,          # 3 × 2.0 — 디노이즈 열림 커널(원본, 홀수)
                    'lane_block_size': 51,     # 25 × 2.0 — 적응형 임계값 창(홀수, 원본)
                    # 가로선(정지선·격자) 제거 — BEV 행 밀도 임계(폭 비율). 끊긴 정지선도 잡음.
                    'lane_horiz_filter_frac': 0.4,
                    'debug_log': False,
                },
            ],
        ),
        Node(
            package='monitor',
            executable='monitor_node',
            name='monitor_node',
            output='screen',
            parameters=[
                {
                    # vehicle_config 에서 IMAGE_TOPIC(/camera/image/compressed),
                    # WEB_HOST/PORT, 5종 토픽·디스플레이 해상도를 로드. 이로써
                    # camera_node → monitor_node(Flask 대시보드) 가 연결된다.
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
        ),
        Node(
            package='inference',
            executable='inference_node',
            name='inference_node',
            output='screen',
            parameters=[
                {
                    'model_path': model_path,
                    'vehicle_config_file': vehicle_config_path,
                    # --- 인식 확정 프레임(반응 지연 직결. YOLO ≈3Hz → 프레임당 ~0.3s) ---
                    # 출발 초록불은 recall 우선(미인식=미션 실패). 먼 신호등이
                    # 간헐 검출돼도 즉시 출발하도록 1프레임 확정.
                    'start_confirm_frames': 1,  # 초록불 출발 (1≈0.3s)
                    # 출발 대기 중 초록불 전용 낮은 신뢰도 임계값(먼 신호등 recall).
                    # 출발 전 초록불에만 적용, 그 외/출발 후엔 conf_threshold(0.25).
                    'green_start_conf': 0.12,
                    # 출발 초록불 HSV 색 폴백(YOLO 보완). 출발 전에만 활성.
                    'green_hsv_fallback': True,
                    'green_hsv_lower': [40, 80, 80],   # OpenCV H 0~180
                    'green_hsv_upper': [90, 255, 255],
                    'green_roi_top_ratio': 0.0,        # 상단 ROI(바닥 배제)
                    'green_roi_bottom_ratio': 0.5,     # 신호등 인식 상단 50% 제한과 일치
                    'green_min_area': 8.0,             # blob 최소 면적(px²)
                    'green_blob_score': 0.5,
                    'confirm_frames': 2,        # 좌/우 표지판 분기 (margin 게이팅이 보호)
                    'stop_confirm_frames': 2,   # 빨간불 정지
                    # 신호등(빨강/초록) 공통 ROI: 박스 세로중심이 프레임 상단 이 비율
                    # 안일 때만 유효(신호등은 트랙 위쪽). 상단 50% 제한.
                    'light_roi_top_ratio': 0.5,
                    # 빨간불 오검출(ArUco 구간 '빨간 바닥') 배제 — 기하 게이팅.
                    # 실제 신호등은 프레임 상단에 작게, 빨간 바닥은 하단에 크게
                    # 잡힌다. 박스 세로중심이 프레임의 이 비율보다 아래면(바닥) 무시.
                    # 카메라 각도에 따라 조정: 신호등이 하단에 잡히면 키우고,
                    # 바닥이 계속 오검출되면 줄인다. /inference/detections 의 box 로 튜닝.
                    'redlight_max_y_ratio': 0.6,
                    # 박스 높이가 프레임의 이 비율 이상이면(너무 큼=바닥) 무시.
                    'redlight_max_h_ratio': 0.5,
                    # 출발 게이트/순항 — 런타임 인자로 노출(라인 트래킹 점검용).
                    'require_green_start': ParameterValue(
                        require_green_start, value_type=bool),
                    # --- 출발 킥스타트: 초록불 확정 직후 조향 없이 직진 출발 ---
                    'start_kick_throttle': 0.2,   # 킥스타트 throttle(정지마찰 극복)
                    # 킥스타트 지속(초). control_hz 로 환산. 0=킥 비활성.
                    'start_kick_seconds': ParameterValue(
                        start_kick_seconds, value_type=float),
                    # --- throttle (트랙 현장 조정 대상) ---
                    'cruise_throttle': ParameterValue(
                        cruise_throttle, value_type=float),  # 직진 순항 (기본 0.175)
                    'corner_throttle': 0.16,  # 코너 감속 throttle
                    # 코너 판정 곡률 임계(정규화 [-1,1] 스케일). 실측 직선 curvature
                    # 노이즈가 ~0.04 이므로 0.30 은 사실상 발동 안 됨 → 0.12 로 낮춰
                    # 실제 커브에서 감속되게 함. ctrl 로그의 curv/corner_hold 로 튜닝.
                    'corner_curvature_threshold': 0.12,
                    # 커브 진입 전 예측 감속 홀드 계수(0~1). 클수록 더 일찍/오래 감속 유지.
                    'curve_hold_decay': 0.85,
                    'turn_throttle': 0.16,     # 갈림길 커밋 중 감속
                    # 조향 중(바퀴 꺾는 중) throttle: |steer-trim| 이 임계 이상이면
                    # 실제 조향각에 반응해 감속(곡률 기반 corner_throttle 과 별개).
                    'steer_throttle': 0.16,
                    'steer_throttle_threshold': 0.05,
                    # --- 조향 (트랙 현장 조정 대상) ---
                    'steer_sign': -1.0,        # 전체 조향 극성(벤치서 반대면 뒤집기)
                    'steer_kp': 0.6,           # 차선 오프셋 비례 게인
                    'steer_kd': 0.15,          # 미분 게인(떨림 억제)
                    # 곡률 피드포워드: 다가오는 커브를 미리 조향(이탈 방지). 직선
                    # curvature 노이즈(~0.04)엔 무영향, 실커브에서만 유효. 실차서
                    # ctrl 로그의 curv 대비 커브 진입 조기성이 부족하면 키운다.
                    'curve_ff': 0.30,
                    'steer_slew': 0.15,        # 프레임당 최대 조향 변화
                    # --- 갈림길 (트랙 현장 조정 대상) ---
                    'turn_bias': 0.7,          # 커밋 중 방향 바이어스 크기(강하게 꺾음)
                    'commit_lane_weight': 0.3,  # 커밋 중 차선 PD 비중(0=차선무시,1=평소)
                    'commit_steer_slew': 0.30,  # 커밋 중 조향 변화 상한(분기 신속완성)
                    'fork_commit_frames': 30,  # 커밋 지속(제어 프레임, 30@20Hz≈1.5s)
                    # 갈림길 커밋 트리거(근접/소실, B.3): 방향은 표지판을 멀리서
                    # 봐도 일찍 래치(기억)하되, 실제 꺾기는 표지판이 가까워졌을 때만
                    # 시작한다(멀리서 조기 분기해 코스 이탈하는 것 방지).
                    # sign_proximity_metric: 근접 지표. 우리 차량은 카메라가 높아
                    #   내려다보고 표지판이 낮게 설치돼 있으므로 'bottom_y'(표지판이
                    #   프레임 아래로 내려오는 정도)가 가장 견고. 대안: 'area'(면적),
                    #   'height'(박스 높이 — 높은 카메라에선 근접 시 포화·눌림이라 비권장).
                    'sign_proximity_metric': ParameterValue(
                        sign_proximity_metric, value_type=str),
                    # sign_commit_ratio: 근접지표(0~1) ≥ 이 값이면 '가까움' → 커밋 시작.
                    #   ⚠️ 실트랙 캘리브레이션 필수 — tools/sign_commit_calibration.py 로
                    #   "이제 꺾어야 한다" 싶은 지점의 지표값(bottom_y=box y2/480)을 재고
                    #   그 값 근처로 잡는다(크면 커밋 지연, 작으면 조기 분기).
                    #   지표를 바꾸면 임계도 다시(bottom_y≈0.55~0.7, area≈0.03~0.10).
                    #   런타임 인자로 노출 → sign_commit_ratio:=0.62 로 재빌드 없이 조정.
                    'sign_commit_ratio': ParameterValue(
                        sign_commit_ratio, value_type=float),
                    # 소실 폴백(백업): 표지판이 프레임 밖으로 벗어날 때 대비. 지표가
                    # sign_lost_min_ratio 이상 커진 뒤 sign_lost_commit_frames(YOLO
                    # 프레임) 연속 미검출이면 커밋. 표지판이 계속 보이면 안 쓰임.
                    'sign_lost_min_ratio': 0.45,
                    'sign_lost_commit_frames': 3,
                    # 출발 직진 유예(제어 프레임): 출발 후 이 구간 동안 표지판 분기
                    # 억제·직진. 출발→S자→갈림길 순서 대응. 0=비활성.
                    'start_straight_frames': ParameterValue(
                        start_straight_frames, value_type=int),
                    # --- 속도 (트랙 현장 조정 대상) ---
                    'lane_lost_throttle': 0.16,
                    # --- ArUco 동적 장애물 정지/재출발 (B.4) ---
                    # 장애물 마커 등장 시 정지, 소멸 시 재출발(정지 중 스탑워치 멈춤).
                    'aruco_enabled': True,
                    'aruco_dict': 'DICT_6X6_50',   # 대회 마커(실측 확정)
                    'aruco_target_ids': [3],       # 규정 마커 ID. []=아무 마커나 인정
                    # 근접 게이팅(면적 비율). 0.0=거리 무관. 먼 마커 오정지 시 키운다.
                    # /inference/detections 로그 대신 ctrl 로그의 aruco_stop 으로 튜닝.
                    'aruco_min_area_ratio': 0.0,
                    # ROI: 마커 중심이 이 정규화 사각형[x0,y0,x1,y1] 안일 때만 정지.
                    # 기본 = 하단 60%·가로 중앙 60%(주행 경로). []=전체 화면.
                    'aruco_roi_norm': [0.2, 0.4, 0.8, 1.0],
                    # 비대칭 히스테리시스(9.5): 정지 진입은 민감(작게), 재출발은
                    # 소멸 확인 후 보수적(크게). 카메라 프레임 rate 단위.
                    'aruco_stop_confirm_frames': 2,
                    'aruco_clear_confirm_frames': 6,
                },
            ],
        ),
    ])