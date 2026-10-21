
import os
import numpy as np

from config import sign_names
from preprocessing import normalize_sequence, resample_sequence


def build_dataset_from_official_keypoints(
    data_path,
    signers,
    frames_per_sequence=40
):
    X, y, meta = [], [], []

    for person in signers:
        person_path = os.path.join(data_path, person)
        if not os.path.isdir(person_path):
            continue

        for sign_id in sorted(os.listdir(person_path)):
            arabic_sign = sign_names[int(sign_id) - 1]

            class_idx = int(sign_id) - 1

            label_path = os.path.join(person_path, sign_id)

            for video in sorted(os.listdir(label_path)):
                video_path = os.path.join(label_path, video)
                if not os.path.isdir(video_path):
                    continue

                lh, rh, pose = [], [], []

                frame_files = sorted([
                    f for f in os.listdir(video_path)
                    if f.endswith(".npy")
                ])

                for frame in frame_files:
                    frame_arr = np.load(
                        os.path.join(video_path, frame)
                    )

                    lh.append(frame_arr[:63])
                    rh.append(frame_arr[63:126])
                    pose.append(frame_arr[126:])

                if len(pose) == 0:
                    continue

                seq = normalize_sequence(lh, rh, pose)

                if np.mean(np.abs(seq[1:] - seq[:-1])) < 1e-3:
                    continue

                seq_resampled = resample_sequence(
                    seq,
                    target_len=frames_per_sequence
                )

                X.append(seq_resampled)
                y.append(class_idx)

                meta.append({
                    "signer": person,
                    "label_dir": sign_id,
                    "video_id": video,
                    "arabic": arabic_sign,
                    "class_idx": class_idx
                })

    X_final = np.asarray(X, dtype=np.float32)
    y_final = np.asarray(y, dtype=np.int32)

    return X_final, y_final, meta
