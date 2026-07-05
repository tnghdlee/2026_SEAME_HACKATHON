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
                    'roi_top': 50,
                    'lane_num_bands': 3,
                    'lane_valid_min_px': 40,
                    'debug_log': False,
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
                    # 코너 판정 곡률 임계(정규화 [-1,1] 스케일). 트랙 curvature 로그로 조정.
                    'corner_curvature_threshold': 0.30,
                    # 커브 진입 전 예측 감속 홀드 계수(0~1). 클수록 더 일찍/오래 감속 유지.
                    'curve_hold_decay': 0.85,
                    'turn_throttle': 0.13,     # 갈림길 커밋 중 감속
                    # --- 조향 (트랙 현장 조정 대상) ---
                    'steer_sign': -1.0,        # 전체 조향 극성(벤치서 반대면 뒤집기)
                    'steer_kp': 0.6,           # 차선 오프셋 비례 게인
                    'steer_kd': 0.15,          # 미분 게인(떨림 억제)
                    'steer_slew': 0.15,        # 프레임당 최대 조향 변화
                    # --- 갈림길 (트랙 현장 조정 대상) ---
                    'turn_bias': 0.35,         # 커밋 중 방향 바이어스 크기
                    'fork_commit_frames': 30,  # 커밋 지속(제어 프레임, 30@20Hz≈1.5s)
                    # --- 속도 (트랙 현장 조정 대상) ---
                    'lane_lost_throttle': 0.10,
                },
            ],
        ),
    ])
