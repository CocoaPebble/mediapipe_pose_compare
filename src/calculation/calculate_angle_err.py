import numpy as np


def angle_err(gt_angles, mp_angles, keypose=None):
    all_err = []
    if keypose:
        gt_frame = gt_angles[keypose]
        for mp_frame in mp_angles:
            err = np.round(np.sum(np.abs(np.array(gt_frame) - np.array(mp_frame))), 3)
            all_err.append(err)
    else:
        frames = min(len(gt_angles), len(mp_angles))
        for frame in range(frames):
            err = np.round(
                np.sum(np.abs(np.array(gt_angles[frame]) - np.array(mp_angles[frame]))),
                3,
            )
            all_err.append(err)
    return all_err


def each_angle_err(
    gt_angles, mp_angles, keypose=-1, actual_min_pos=None, start=None, end=None
):
    if keypose < -1:
        print("need keypose")
        raise TypeError
    if not start:
        start = actual_min_pos
    if not end:
        end = actual_min_pos

    all_err = []
    for frame in range(start, end + 1):
        err = np.abs(np.array(gt_angles[keypose]) - np.array(mp_angles[frame]))
        err = np.round(err, 2)
        print("At frame", frame, "angle errs are", err)
        all_err.append(err.tolist())

    return all_err


def convert_err_to_plot(err, start_frame):
    # err = err.tolist()
    new_arr = []
    for i, ele in enumerate(err):
        arr = [start_frame + i] + ele
        new_arr.append(arr)

    return new_arr
