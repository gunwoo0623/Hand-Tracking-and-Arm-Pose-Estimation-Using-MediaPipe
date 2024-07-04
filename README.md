# Hand Tracking and Arm Pose Estimation Using a Camera

This project was undertaken as an open project and completed within a short period due to numerous assignments and ongoing studies. The primary aim was to explore the feasibility of using a camera as a control interface for the manipulator. The project utilized MediaPipe. However, further work is needed to obtain the angles of the wrist and shoulder. Additionally, implementing machine learning techniques are essential to enhance accuracy.

## Demonstration Video

### Hand Tracking and Arm Pose Estimation

The following video demonstrates the system's accuracy in capturing hand and upper body movements, identifying whether fists are closed or open, and measuring elbow angles. It also shows the ability to pause and capture specific moments for detailed pose analysis:

https://github.com/gunwoo0623/Hand-Tracking-and-Arm-Pose-Estimation-Using-MediaPipe/assets/52570227/f0ec1ffd-bfdf-42c4-8bd0-7cdd52a9bff8


## Development of Hand Tracking and Arm Pose Estimation

This project utilizes [MediaPipe Hands](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md) and [Holistic](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md), an open-source framework developed by Google. MediaPipe tracks 21 hand landmarks and 33 key body points, making it ideal for precise gesture detection and comprehensive full-body motion analysis.

<img src="https://github.com/gunwoo0623/Hand-Tracking-and-Arm-Pose-Estimation-Using-MediaPipe/assets/52570227/6f591d43-b088-45b2-8e83-d26e87cca1a7" width="50%">
<img src="https://github.com/gunwoo0623/Hand-Tracking-and-Arm-Pose-Estimation-Using-MediaPipe/assets/52570227/044cb669-c340-41b1-93a8-1cadfdfe81bb" width="50%">

### Software Setup

The development environment for this project utilized PyCharm Community Edition 2023.3.4. The following settings and libraries were essential:
* Python: 3.8.0
* mediapipe: 0.10.11
* opencv-python: 4.9.0 or later
* numpy: 1.24.3 or later

### Flowchart

This flowchart outlines the system's operations, from initialization and setup to real-time frame capture, applying MediaPipe for gesture recognition, and displaying processed frames with FPS data. It shows how the system handles inputs and navigates decision points.

<img src="https://github.com/gunwoo0623/Hand-Tracking-and-Arm-Pose-Estimation-Using-MediaPipe/assets/52570227/15f12ae4-d3fb-40cb-8487-08d9401e9bd4" width="70%">

### Results

Visual data captured during the project highlight two distinct gestures: closed fists and open hands. MediaPipe demonstrates high accuracy in distinguishing between these hand configurations, effectively detecting subtle variations in gestures.

<div class="form-group">
        <div style="height: 200px; width: 500px;">
                <img style="height: 100%; width: 35%; float:left;" src="https://github.com/gunwoo0623/Hand-Tracking-and-Arm-Pose-Estimation-Using-MediaPipe/assets/52570227/1da4420b-a483-4afd-bd87-fc1d0bbe7f89">
                <img style="height: 100%; width: 35%; float:right;" src="https://github.com/gunwoo0623/Hand-Tracking-and-Arm-Pose-Estimation-Using-MediaPipe/assets/52570227/cc8c46dd-51bd-4176-83ed-b3bbadd25d6f">
        </div>
</div>

The analysis reveals that tracking two hands simultaneously imposes a computational load, leading to a noticeable reduction in frames per second (FPS). This reduction is primarily due to the increased complexity of tracking multiple landmarks in real-time.

<div class="form-group">
        <div style="height: 200px; width: 500px;">
                <img style="height: 100%; width: 35%; float:left;" src="https://github.com/gunwoo0623/Hand-Tracking-and-Arm-Pose-Estimation-Using-MediaPipe/assets/52570227/8a7b37e5-83e9-4ce3-af12-9a87c3369d29">
                <img style="height: 100%; width: 35%; float:right;" src="https://github.com/gunwoo0623/Hand-Tracking-and-Arm-Pose-Estimation-Using-MediaPipe/assets/52570227/ca8c33ca-2bb4-417d-b2d6-0c69fd1f22e5">
        </div>
</div>

These images depict detailed 3D plots of hand landmarks and full-body poses, illustrating the joint positions and limb orientations in three-dimensional space.

## Future Work

### Model Changes

The video below highlights future work should prioritize incorporating additional landmarks, particularly those adjacent to landmark 0 on the hand and around landmarks 11, 12, 13, and 14 on the body, to accurately obtain wrist and shoulder angles.

https://github.com/gunwoo0623/Hand-Tracking-and-Arm-Pose-Estimation-Using-MediaPipe/assets/52570227/56f6d939-9226-426f-9d1a-d151efa9f759

### Machine Learning Algorithm

kNN MLP SVM
