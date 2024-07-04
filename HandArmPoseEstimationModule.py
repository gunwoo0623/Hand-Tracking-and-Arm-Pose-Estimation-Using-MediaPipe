import cv2  # OpenCV 라이브러리 import
import mediapipe as mp  # MediaPipe 패키지 import하고 mp라는 별칭으로 사용하겠다는 뜻.
import math  # math 모듈 import
import time
import copy
import numpy as np

from mediapipe.python.solutions.drawing_utils import DrawingSpec


# 거리 계산 함수 선언
def distance(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))  # 두 점 p1, p2의 x, y 좌표로 거리를 계산한다.


# 3D 형체 x-좌표 축을 기준으로 뒤집기 함수 선언
def plot_flipped_landmarks(hand_world_landmarks, connections):
    flipped_landmarks = copy.deepcopy(hand_world_landmarks)
    for landmark in flipped_landmarks.landmark:
        landmark.x = -landmark.x  # x-좌표 뒤집기
    mp_drawing.plot_landmarks(flipped_landmarks, connections, azimuth=180)


# 필요한 landmarks만 고려하는 함수 선언
def filter_landmarks(landmarks, custom_pose_landmarks):
    filtered_landmarks = copy.deepcopy(landmarks)
    for idx, landmark in enumerate(filtered_landmarks.landmark):
        if idx not in custom_pose_landmarks:
            # landmark 모두 0으로 리셋
            landmark.x = 0
            landmark.y = 0
            landmark.z = 0
            landmark.visibility = 0
    return filtered_landmarks


# 3개의 점으로 각도 구하는 함수 선언
def calculate_angle(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def is_valid(landmark):
    x, y = landmark
    return 0 < x < 1 and 0 < y < 1


# MediaPipe 패키지에서 사용할 기능들.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands  # 손 인식을 위한 객체
mp_pose = mp.solutions.pose  # 몸 인식을 위한 객체

default_hands_connections = mp_hands.HAND_CONNECTIONS  # 기존 손 모양을 default로 정의
custom_hands_connections = list(default_hands_connections)  # custom 생성 후 default 내용 저장
custom_hands_connections.append((5, 2))  # custom 내에 5번에서 2번 사이를 선 추가

custom_pose_landmarks = {11, 12, 13, 14, 15, 16}  # 상체만 보일 예정이므로 6개의 landmarks만 선정
custom_pose_connections = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]

# 기본적인 카메라 세팅
# (VideoCapture Class 참고: https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a8c6d8c2d37505b5ca61ffd4bb54e9a7c)
cap = cv2.VideoCapture(0)  # 비디오 캡처 객체 생성
# (Enumrations 참고: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d)
frame_width, frame_height = 640, 480  # 카메라 화질에 따라 달라짐. 이는 .shape 기능으로도 알 수 있음
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)  # 현재 노트북 내장 웹캠의 최대 화질 (640:480) 및 비율 4:3
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# 현재 시간과 과거 시간 리셋
cur_time = 0
pre_time = 0

# 선명한 시각효과를 위한 Dual-Line 정의
black_spec = DrawingSpec(color=(0, 0, 0), thickness=4, circle_radius=3)
white_spec = DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)

# Hands 모듈 설정 (참고: https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)
hands = mp_hands.Hands(
    static_image_mode=True,  # False이면 손이 트랙에서 벗어나면 탐지/True이면 매 순간 손을 탐지
    max_num_hands=2,  # 최대 탐지할 손의 갯수
    min_detection_confidence=0.5,  # 50% 이상 손임을 확신하면 손을 탐지하기
    # min_tracking_confidence=0.5  # static_image_mode가 true이면 무시
)  # 손 인식 객체 생성

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,  # 정확도와 추론 시간 비례 (0, 1, 2)
    # smooth_landmarks=True,  # static_image_mode가 true이면 무시
    # enable_segmentation=True,  # 세그멘테이션 제공
    # smooth_segmentation=True,  # static_image_mode이 True 이거나 enable_segmentation이 False이면 무시
    min_detection_confidence=0.5,  # 50% 이상 몸임을 확신하면 몸을 탐지하기
    # min_tracking_confidence=0.5  # static_image_mode가 true이면 무시
)

while cap.isOpened():  # 카메라가 커져있을 때까지 실행
    res, frame = cap.read()  # 카메라 데이터 읽기

    if not res:  # 프레임 읽었는지 확인
        print("Failed to read the frame from the camera.")
        break  # 반복문 종료

    mirrored_frame = cv2.flip(frame, 1)  # 셀프 카메라처럼 좌우 반전
    image = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)  # 미디어파이프에서 인식 가능한 색공간으로 변경
    results_hands = hands.process(image)  # 이미지에서 손을 찾고 결과를 반환
    results_pose = pose.process(image)  # 이미지에서 몸을 찾고 결과를 반환

    # Frame Per Second(FPS) 확인하기
    cur_time = time.time()  # 현재 시간
    fps = 1 / (cur_time - pre_time)
    pre_time = cur_time
    cv2.putText(mirrored_frame, "FPS : {:.2f}".format(float(fps)), (10, 40), cv2.FONT_HERSHEY_TRIPLEX,
                1, (0, 0, 0), 4, cv2.LINE_AA)  # 선명함을 위한 바깥 검은 테두리
    cv2.putText(mirrored_frame, "FPS : {:.2f}".format(float(fps)), (10, 40), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)  # 메인 FPS 숫자

    # 1. 손을 표시하기
    if results_hands.multi_hand_landmarks:  # 손이 인식되었는지 확인
        for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks,
                                              results_hands.multi_handedness):  # 반복문을 활용해 인식된 손의 주요 부분을 그림으로 그려 표현
            mp_drawing.draw_landmarks(
                mirrored_frame,
                hand_landmarks,
                custom_hands_connections,
                black_spec,
                black_spec
            )

            mp_drawing.draw_landmarks(
                mirrored_frame,
                hand_landmarks,
                custom_hands_connections,
                white_spec,
                white_spec
            )

            # 각각의 landmark 값 저장
            x_coords_hands = [landmark.x * frame_width for landmark in hand_landmarks.landmark]
            y_coords_hands = [landmark.y * frame_height for landmark in hand_landmarks.landmark]

            # 최대 최소값을 구한 뒤 박스 구성
            x_min, x_max, y_min, y_max = min(x_coords_hands), max(x_coords_hands), min(y_coords_hands), max(
                y_coords_hands)
            cv2.rectangle(mirrored_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(mirrored_frame, (int(x_min), int(y_min)), (int(x_max), int(y_min - 22)), (0, 0, 0), -1,
                          cv2.LINE_AA)

            info_handedness = handedness.classification[0].label[0:]

            # 엄지손가락부터 새끼손가락까지 손가락이 펴졌는지 확인한다.
            points = hand_landmarks.landmark  # landmark 좌표 정보들을 points라는 변수로 활용
            fingers = [0, 0, 0, 0, 0]  # 편 손가락을 확인하기 위한 변수, 엄지손가락 ~ 새끼손가락 순서로 값을 확인한다.
            if distance(points[4], points[9]) > distance(points[3], points[9]):  # 엄지손가락 확인하기
                fingers[0] = 1  # 폈으면 fingers에 1을 할당한다.
            for i in range(1, 5):  # 검지 ~ 새끼손가락 순서로 확인한다.
                if distance(points[4 * (i + 1)], points[0]) > distance(points[4 * (i + 1) - 1], points[0]):
                    fingers[i] = 1  # 폈으면 해당하는 손가락 fingers[i]에 1을 할당한다.

            # 펴진 손가락의 개수에 따라 모양을 인식하고 이미지에 출력한다.
            if fingers == [0, 0, 0, 0, 0]:  # 손가락이 모두 접힌 경우
                hand_shape = "Close"  # 그리퍼 잡기
            elif fingers == [1, 1, 1, 1, 1]:  # 손가락이 5개 모두 펴진 경우
                hand_shape = "Open"  # 그리퍼 펴기
            else:
                hand_shape = ""  # 내용을 출력하지 않음
            info_text = info_handedness + ':' + hand_shape
            cv2.putText(mirrored_frame, hand_shape, (int(x_min + 5), int(y_min - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. 몸을 표시하기
    if results_pose.pose_landmarks:
        filter_pose_landmarks = filter_landmarks(results_pose.pose_landmarks, custom_pose_landmarks)
        mp_drawing.draw_landmarks(
            mirrored_frame,
            filter_pose_landmarks,
            custom_pose_connections,
            black_spec,
            black_spec,
        )

        mp_drawing.draw_landmarks(
            mirrored_frame,
            filter_pose_landmarks,
            custom_pose_connections,
            white_spec,
            white_spec
        )

        # Pose에서 도출될 수 있는 각도 구하기
        landmarks_index = results_pose.pose_landmarks.landmark  # landmark 인덱스 정의

        # 필요한 신체부위의 위치 얻기
        left_shoulder = [landmarks_index[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks_index[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks_index[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks_index[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_elbow = [landmarks_index[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks_index[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        right_elbow = [landmarks_index[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks_index[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        left_wrist = [landmarks_index[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks_index[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks_index[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks_index[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # 바운딩 박스 만들기 위한 구성
        x_coords_pose_left = [left_shoulder[0], left_elbow[0], left_wrist[0]]
        y_coords_pose_left = [left_shoulder[1], left_elbow[1], left_wrist[1]]
        x_min_left, x_max_left = min(x_coords_pose_left) * frame_width, max(x_coords_pose_left) * frame_width
        y_min_left, y_max_left = min(y_coords_pose_left) * frame_height, max(y_coords_pose_left) * frame_height

        x_coords_pose_right = [right_shoulder[0], right_elbow[0], right_wrist[0]]
        y_coords_pose_right = [right_shoulder[1], right_elbow[1], right_wrist[1]]
        x_min_right, x_max_right = min(x_coords_pose_right) * frame_width, max(x_coords_pose_right) * frame_width
        y_min_right, y_max_right = min(y_coords_pose_right) * frame_height, max(y_coords_pose_right) * frame_height

        # 팔꿈치 각도 구하기
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # 바운딩 박스 만들기 및 팔꿈치 각도 표시
        if is_valid(left_shoulder) and is_valid(left_elbow) and is_valid(left_wrist):
            cv2.rectangle(mirrored_frame, (int(x_min_left), int(y_min_left)), (int(x_max_left), int(y_max_left)),
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(mirrored_frame, (int(x_min_left), int(y_max_left)), (int(x_max_left), int(y_max_left + 22)),
                        (0, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(mirrored_frame, "Elbow: {:.2f}".format(left_elbow_angle), (int(x_min_left + 5), int(y_max_left + 22 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if is_valid(right_shoulder) and is_valid(right_elbow) and is_valid(right_wrist):
            cv2.rectangle(mirrored_frame, (int(x_min_right), int(y_min_right)), (int(x_max_right), int(y_max_right)),
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(mirrored_frame, (int(x_min_right), int(y_max_right)), (int(x_max_right), int(y_max_right + 22)),
                        (0, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(mirrored_frame, "Elbow: {:.2f}".format(right_elbow_angle), (int(x_min_right + 5), int(y_max_right + 22 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Hands Tracking and Arm Pose Estimation", mirrored_frame)  # 영상을 화면에 출력.

    key = cv2.waitKey(1) & 0xFF  # 키보드 입력받기 & 0xFF는 16진법을 의미
    if key == 27:  # ESC를 눌렀을 경우
        break  # 반복문 종료
    elif key == 13:  # Enter를 눌렀을 경우
        if not results_hands.multi_hand_world_landmarks:  # 손이 감지되지 않으면 통과
            continue  # 다시 반복문으로 되돌아가기
        for hand_world_landmarks in results_hands.multi_hand_world_landmarks:  # 감지되면 3D 손 모양 그리기
            plot_flipped_landmarks(hand_world_landmarks, custom_hands_connections)
        plot_flipped_landmarks(filter_pose_landmarks, custom_pose_connections)

cv2.destroyAllWindows()  # 영상 창 닫기
cap.release()  # 비디오 캡처 객체 해제
