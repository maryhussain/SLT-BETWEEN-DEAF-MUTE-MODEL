
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATASET_NUMPY_DIR
from src.config import sign_names
from src.preprocessing import normalize_sequence, resample_sequence


def build_dataset_from_official_keypoints(
    data_path,
    signers,
    frames_per_sequence=40
):
    X, y, meta = [], [], []

    for person in signers:
        person_path = data_path / person

        if not person_path.is_dir():
            continue

        for sign_id in sorted(
            [p.name for p in person_path.iterdir() if p.is_dir()]
        ):
            arabic_sign = sign_names[int(sign_id) - 1]
            class_idx = int(sign_id) - 1

            label_path = person_path / sign_id

            for video in sorted(label_path.iterdir()):
                if not video.is_dir():
                    continue

                lh, rh, pose = [], [], []

                frame_files = sorted(video.glob("*.npy"))

                for frame in frame_files:
                    frame_arr = np.load(frame)

                    lh.append(frame_arr[:63])
                    rh.append(frame_arr[63:126])
                    pose.append(frame_arr[126:])

                if len(pose) == 0:
                    continue

                seq = normalize_sequence(lh, rh, pose)

                if len(seq) < 2:
                    continue

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
                    "video_id": video.name,
                    "arabic": arabic_sign,
                    "class_idx": class_idx
                })

    X_final = np.asarray(X, dtype=np.float32)
    y_final = np.asarray(y, dtype=np.int32)

    return X_final, y_final, meta


def save_datasets():
    data_train_path = DATASET_NUMPY_DIR / "train"
    data_test_path = DATASET_NUMPY_DIR / "test"

    X_train, y_train, meta_train = build_dataset_from_official_keypoints(
        data_path=data_train_path,
        signers=["person01", "person02"]
    )

    X_test, y_test, meta_test = build_dataset_from_official_keypoints(
        data_path=data_test_path,
        signers=["person03"]
    )

    np.save(DATASET_NUMPY_DIR / "X_train.npy", X_train)
    np.save(DATASET_NUMPY_DIR / "y_train.npy", y_train)
    np.save(DATASET_NUMPY_DIR / "X_test.npy", X_test)
    np.save(DATASET_NUMPY_DIR / "y_test.npy", y_test)

    pd.DataFrame(meta_train).to_csv(
        DATASET_NUMPY_DIR / "meta_train.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(meta_test).to_csv(
        DATASET_NUMPY_DIR / "meta_test.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    return X_train, y_train, X_test, y_test


def split_training_data(X_train, y_train, validation_size=0.15):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=42,
        stratify=y_train
    )

    np.save(DATASET_NUMPY_DIR / "x_train.npy", X_tr)
    np.save(DATASET_NUMPY_DIR / "y_train.npy", y_tr)
    np.save(DATASET_NUMPY_DIR / "X_val.npy", X_val)
    np.save(DATASET_NUMPY_DIR / "y_val.npy", y_val)

    print("Training:", X_tr.shape, y_tr.shape)
    print("Validation:", X_val.shape, y_val.shape)

    return X_tr, X_val, y_tr, y_val


def main():
    DATASET_NUMPY_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = save_datasets()

    split_training_data(X_train, y_train)


if __name__ == "__main__":
    main()
