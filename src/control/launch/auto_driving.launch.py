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
    # 순항 0.15). 점검 시 require_green_start:=false cruise_throttle:=0.0 로 주면
    # 바퀴는 안 굴러가고 조향만 차선에 반응한다.
    require_green_start = LaunchConfiguration('require_green_start')
    cruise_throttle = LaunchConfiguration('cruise_throttle')
    # 차선 트랙 프로파일. 기본은 대회 규정 트랙(검은 바닥+양쪽 흰 경계선):
    # brightness/light/split_lanes=True. 연습 트랙(회색 바닥+주황 라인)에서
    # 실주행 테스트할 때만 lane_profile:=orange_track 으로 전환.
    lane_profile = LaunchConfiguration('lane_profile')
    # 출발 직진 유예(제어 프레임): 초록불 출발 확정 후 이 프레임 수 동안 표지판
    # 분기를 억제하고 직진(차선 추종)한다. 실코스가 출발→S자→갈림길 순이라
    # 출발 직후 (오)검출로 즉시 꺾이는 것을 막는다. 20Hz 기준 100≈5s.
    # 실제 갈림길까지 걸리는 시간에 맞춰 튜닝(짧으면 조기 분기, 길면 분기 놓침).
    start_straight_frames = LaunchConfiguration('start_straight_frames')

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
            default_value='0.18',
            description='직진 순항 throttle. 조향만 점검하려면 0.0 으로 주면 바퀴가 안 돈다.',
        ),
        DeclareLaunchArgument(
            'lane_profile',
            default_value='white_track',
            description=('차선 트랙 프로파일. 기본 white_track(대회: 검은 바닥+흰 '
                         '경계선, brightness/light/split). 연습 트랙은 orange_track.'),
        ),
        DeclareLaunchArgument(
            'start_straight_frames',
            default_value='100',
            description=('출발 직진 유예(제어 프레임, 20Hz 기준 100≈5s). 초록불 출발 '
                         '후 이 구간 동안 표지판 분기를 억제하고 직진한다. 0=비활성.'),
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
                    # 트랙 프로파일이 검출 방식(method/polarity/split_lanes)을 결정.
                    # 기본 white_track(대회). 개별 검출 param 을 명시하면 프리셋을
                    # 덮어쓴다(opencv_node 참조). 여기선 프로파일에 맡기고,
                    # 공통 기하(ROI·밴드)만 명시 — 실트랙 튜닝 대상.
                    'lane_profile': lane_profile,
                    # 아래 픽셀 기준 값들은 발행 해상도 800×600 기준(vehicle_config
                    # IMAGE_WIDTH/HEIGHT). 320×240 기준값에서 선형 ×2.5, 면적
                    # ×6.25 로 스케일했다. 해상도 변경 시 함께 재조정할 것.
                    'roi_top': 125,           # 50 × (600/240) — 하단 ROI 시작 Y
                    'lane_num_bands': 3,      # 해상도 무관
                    'lane_valid_min_px': 250,  # 40 × 6.25 — 밴드 유효 픽셀 하한
                    'morph_ksize': 7,          # 3 × 2.5 — 디노이즈 열림 커널
                    'lane_block_size': 63,     # 25 × 2.5 — 적응형 임계값 창(홀수)
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
                    'start_confirm_frames': 2,  # 초록불 출발 (2≈0.6s, 1≈0.3s)
                    'confirm_frames': 2,        # 좌/우 표지판 분기 (margin 게이팅이 보호)
                    'stop_confirm_frames': 2,   # 빨간불 정지
                    # 출발 게이트/순항 — 런타임 인자로 노출(라인 트래킹 점검용).
                    'require_green_start': ParameterValue(
                        require_green_start, value_type=bool),
                    # --- throttle (트랙 현장 조정 대상) ---
                    'cruise_throttle': ParameterValue(
                        cruise_throttle, value_type=float),  # 직진 순항 (기본 0.2)
                    'corner_throttle': 0.17,   # 코너 감속 throttle
                    # 코너 판정 곡률 임계(정규화 [-1,1] 스케일). 실측 직선 curvature
                    # 노이즈가 ~0.04 이므로 0.30 은 사실상 발동 안 됨 → 0.12 로 낮춰
                    # 실제 커브에서 감속되게 함. ctrl 로그의 curv/corner_hold 로 튜닝.
                    'corner_curvature_threshold': 0.12,
                    # 커브 진입 전 예측 감속 홀드 계수(0~1). 클수록 더 일찍/오래 감속 유지.
                    'curve_hold_decay': 0.85,
                    'turn_throttle': 0.13,     # 갈림길 커밋 중 감속
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
                    # 출발 직진 유예(제어 프레임): 출발 후 이 구간 동안 표지판 분기
                    # 억제·직진. 출발→S자→갈림길 순서 대응. 0=비활성.
                    'start_straight_frames': ParameterValue(
                        start_straight_frames, value_type=int),
                    # --- 속도 (트랙 현장 조정 대상) ---
                    'lane_lost_throttle': 0.10,
                },
            ],
        ),
    ])