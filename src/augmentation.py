import numpy as np
from config import sign_names
from src.augmentation import augment_data_smart
from src.config import DATASET_NUMPY_DIR
from src.preprocessing import resample_sequence

# 1. Gaussian Noise
def add_noise(seq, noise_level=0.01):
    noise = np.random.normal(0, noise_level, seq.shape)
    return seq + noise

# 2. Scaling
def scale(seq, factor_range=(0.97, 1.03)):
    factor = np.random.uniform(*factor_range)
    return seq * factor

# 3. Translation
def translate(seq, shift_range=(-0.01, 0.01)):
    shift = np.random.uniform(*shift_range)
    return seq + shift

# 4. Speed up
def speed_up(seq):
    return seq[::2]

# 5. Slow down
def slow_down(seq):
    return np.repeat(seq, 2, axis=0)

# 6. Frame dropping
def drop_frames(seq, drop_rate=0.1):
    mask = np.random.rand(len(seq)) > drop_rate
    if np.sum(mask) < 5:
        return seq  
    return seq[mask]


def augment_data_smart(X, y):
    X_aug, y_aug = [], []
    target_len = X.shape[1]

    for i in range(len(X)):
        seq = X[i]
        label = y[i]
        label_name =sign_names[label]

        X_aug.append(seq)
        y_aug.append(label)

        if label_name in ["شكراً لحسن استماعكم", "أريد تشغيل المدفأة", "أريد الذهاب إلى النوم",
    ]:
            n_aug = 4
        else:
            n_aug = 1

        for _ in range(n_aug):
            new_seq = seq.copy()

            
            choice = np.random.choice(
                        ["noise", "scale", "translate", "temporal", "drop"],
                        p=[0.35, 0.2, 0.15, 0.2, 0.1]
                    )

            if choice == "noise":
                new_seq = add_noise(new_seq, noise_level=0.01)
                

            elif choice == "scale":
                factor = np.random.uniform(0.95, 1.05)
                new_seq = new_seq * factor

            elif choice == "translate":
                shift = np.random.uniform(-0.02, 0.02)
                new_seq = new_seq + shift

            elif choice == "temporal":
                if np.random.rand() > 0.5:
                    new_seq = new_seq[::2]  
                else:
                    new_seq = np.repeat(new_seq, 2, axis=0)  

            elif choice == "drop":
                mask = np.random.rand(len(new_seq)) > 0.1
                new_seq = new_seq[mask]

            new_seq = resample_sequence(new_seq, target_len)

            X_aug.append(new_seq)
            y_aug.append(label)

    return np.array(X_aug, dtype=np.float32), np.array(y_aug)



def main():
    X_train = np.load(DATASET_NUMPY_DIR / "x_train.npy")
    y_train = np.load(DATASET_NUMPY_DIR / "y_train.npy")

    X_train_aug, y_train_aug = augment_data_smart(X_train, y_train)

    np.save(DATASET_NUMPY_DIR / "x_train_aug.npy", X_train_aug)
    np.save(DATASET_NUMPY_DIR / "y_train_aug.npy", y_train_aug)

    print("Augmented training data:", X_train_aug.shape)
    print("Augmented training labels:", y_train_aug.shape)


if __name__ == "__main__":
    main()