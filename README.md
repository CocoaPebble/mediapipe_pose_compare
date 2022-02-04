# mediapipe_pose_compare
Joint angle comparison of mediapipe prediction results bvh conversion with ground truth bvh

## Through video
The bvh joint angle formed by the 3D joint points predicted by Mediapipe through the video is compared with the bvh joint angle of the model in AMASS, and the prediction result is evaluated by frame-by-frame comparison.

1. Predict with mp pose to get 3D joints position
2. Map 3D joints to gt 17 points
3. Convert 3D joints position to npy data
4. Process npy data to bvh format
5. Calculate sum of joint angle error frame-by-frame

## Through camera
Mediapipe uses the bvh joint angle formed by the 3D joint points predicted by the camera, and one or several bvh joint angles of the model in AMASS to evaluate the prediction result