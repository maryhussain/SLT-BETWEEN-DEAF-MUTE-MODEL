import os 
import cv2
import mediapipe as mp
import numpy as np
from config import DATASET_VIDEOS_DIR, DATASET_NUMPY_DIR

# base_path
base_path = DATASET_VIDEOS_DIR
# csv_file
output_path = DATASET_NUMPY_DIR
mp_holistic = mp.solutions.holistic

# Detection landmarks
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image, results 

# extract keypoint
def extract_keypoint(results):
    pose = np.array([[res.x, res.y ,res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    left_hand = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten()  if results.left_hand_landmarks else np.zeros(21 * 3)
    right_hand = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten()  if results.right_hand_landmarks else np.zeros(21 * 3)
                          
    return np.concatenate([left_hand,right_hand,pose])


def main():

    with mp_holistic.Holistic(
        min_detection_confidence=0.05,
        min_tracking_confidence=0.05
    ) as holistic:

        # person
        for person in os.listdir(base_path):
            person_path = os.path.join(base_path, person)

            # sign
            for label_name in os.listdir(person_path):
                label_path = os.path.join(person_path, label_name)

                # video
                for video in os.listdir(label_path):
                    video_path = os.path.join(label_path, video)

                    if not os.path.isdir(video_path):
                        continue

                    points = []
                    all_frames = sorted([
                        f for f in os.listdir(video_path)
                        if f.endswith(('.jpg', '.png'))
                    ])

                    # frames
                    for frame in all_frames:
                        frame_path = os.path.join(video_path, frame)
                        img = cv2.imread(frame_path)

                        if img is None:
                            print(f"Error reading image: {frame_path}")
                            continue

                        image, results = mediapipe_detection(img, holistic)
                        keypoints = extract_keypoint(results)

                        print(f"Frame: {frame} | Shape: {keypoints.shape}")

                        frame_name = os.path.splitext(frame)[0]

                        frame_output_dir = os.path.join(
                            output_path, person, label_name, video
                        )
                        os.makedirs(frame_output_dir, exist_ok=True)

                        frame_npy_path = os.path.join(
                            frame_output_dir, f"{frame_name}.npy"
                        )

                        np.save(frame_npy_path, keypoints)


if __name__ == "__main__":
    main()
