import numpy as np 



def normalize_single_frame(lh_flat, rh_flat, pose_flat):

    pose = pose_flat.reshape(-1, 3).copy()
    left_hand = lh_flat.reshape(-1, 3).copy()
    right_hand = rh_flat.reshape(-1, 3).copy()

    left_shoulder = pose[11]
    right_shoulder = pose[12]
    shoulders_valid = np.linalg.norm(left_shoulder)> 0 and np.linalg.norm(right_shoulder)> 0 

    if shoulders_valid:
        pose_center = (left_shoulder + right_shoulder) / 2.0
        pose_scale = np.linalg.norm(left_shoulder - right_shoulder)

    else:
   
        nose = pose[0]
        if np.linalg.norm(nose) > 0:
            pose_center = nose
            pose_scale = 1.0
        else:
            valid_points = pose[np.linalg.norm(pose, axis=1) > 0]
            if len(valid_points) > 0:
                pose_center = np.mean(valid_points, axis=0)
                pose_scale = 1.0
            else:
                pose_center = np.zeros(3)
                pose_scale = 1.0

    pose = pose - pose_center
    pose = pose / max(float(pose_scale), 1e-6)

    def norm_hand(hand):
        wrist = hand[0].copy()
        if np.linalg.norm(wrist) > 0:
            hand = hand - wrist
            palm_ref = hand[9]
            scale = np.linalg.norm(palm_ref)
            if scale < 1e-6 :
                scale = 1.0
            hand = hand / max(float(scale), 1e-6)
        return hand

    lh = norm_hand(left_hand)
    rh = norm_hand(right_hand)

    return np.concatenate([lh.reshape(-1), rh.reshape(-1),pose.reshape(-1)], axis=0).astype(np.float32)


def normalize_sequence(lh_seq, rh_seq,pose_seq):
    frames = []
    min_len = min(len(lh_seq), len(rh_seq) ,len(pose_seq))
    for i in range(min_len):
        normalized_f = normalize_single_frame(lh_seq[i], rh_seq[i], pose_seq[i])
        frames.append(normalized_f)
    return np.asarray(frames, dtype=np.float32)


def resample_sequence(sequence, target_len=40):
    sequence = np.array(sequence)
    n = len(sequence)

    if n == target_len:
        return sequence

    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)

    resampled = np.zeros((target_len, sequence.shape[1]))

    for i in range(sequence.shape[1]):
        resampled[:, i] = np.interp(x_new, x_old, sequence[:, i])

    return resampled.astype(np.float32)