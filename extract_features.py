import numpy as np
import os
import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Dataset Paths
train_folder = "dataset/Train"  
test_folder = "dataset/Test"  

# Extract features function
def extract_pose_hand_landmarks(video_path, max_frames=100, min_frames=30):
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine number of frames to extract (adaptive)
    num_frames = min(max_frames, max(min_frames, video_length // 10))  
    skip_frames = max(video_length // num_frames, 1)  

    landmarks_list = []
    frame_count = 0

    if not cap.isOpened():
        print(f"⚠ Error opening {video_path}")
        return None  

    success, frame = cap.read()
    while success:
        if frame_count % skip_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            frame_landmarks = []

            # Pose (33 landmarks)
            if results.pose_landmarks:
                frame_landmarks.extend([lm.x for lm in results.pose_landmarks.landmark] +
                                       [lm.y for lm in results.pose_landmarks.landmark] +
                                       [lm.z for lm in results.pose_landmarks.landmark])
            else:
                frame_landmarks.extend([0] * (33 * 3))

            # Left Hand (21 landmarks)
            if results.left_hand_landmarks:
                frame_landmarks.extend([lm.x for lm in results.left_hand_landmarks.landmark] +
                                       [lm.y for lm in results.left_hand_landmarks.landmark] +
                                       [lm.z for lm in results.left_hand_landmarks.landmark])
            else:
                frame_landmarks.extend([0] * (21 * 3))

            # Right Hand (21 landmarks)
            if results.right_hand_landmarks:
                frame_landmarks.extend([lm.x for lm in results.right_hand_landmarks.landmark] +
                                       [lm.y for lm in results.right_hand_landmarks.landmark] +
                                       [lm.z for lm in results.right_hand_landmarks.landmark])
            else:
                frame_landmarks.extend([0] * (21 * 3))

            if len(frame_landmarks) == 225:
                landmarks_list.append(frame_landmarks)

        success, frame = cap.read()
        frame_count += 1
        if len(landmarks_list) >= num_frames:
            break

    cap.release()

    # **Fix:** Ensure all videos have exactly `max_frames` frames
    if len(landmarks_list) < max_frames:
        padding = [[0] * 225] * (max_frames - len(landmarks_list))
        landmarks_list.extend(padding)  # Pad with zeros
    elif len(landmarks_list) > max_frames:
        landmarks_list = landmarks_list[:max_frames]  # Trim excess

    return np.array(landmarks_list) if landmarks_list else None

# Process dataset
def process_dataset(folder_path):
    all_landmarks, all_labels = [], []
    class_names = sorted(os.listdir(folder_path))

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(folder_path, class_name)
        video_files = os.listdir(class_path)

        print(f"📂 Processing {len(video_files)} videos in class {class_name}")

        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)
            landmarks = extract_pose_hand_landmarks(video_path)

            if landmarks is None:
                print(f"⚠ No pose extracted from {video_file}")
                continue  

            all_landmarks.append(landmarks)
            all_labels.append(class_idx)

    return np.array(all_landmarks), np.array(all_labels)

# Extract features from Train and Test datasets
X_train, y_train = process_dataset(train_folder)
X_test, y_test = process_dataset(test_folder)

# Save extracted features
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print(f"✅ Train samples: {len(X_train)}, Labels: {len(y_train)}")
print(f"✅ Test samples: {len(X_test)}, Labels: {len(y_test)}")