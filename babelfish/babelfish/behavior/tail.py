import numpy as np
from numba import jit

# https://github.com/portugueslab/example_stytra_analysis/blob/230fd312ecb15d353170c62a258d44dedb88aa11/utilities.py#L244
@jit(nopython=True)
def extract_segments_above_thresh(
    vel, threshold=0.1, min_duration=20, pad_before=12, pad_after=25, skip_nan=True
):
    """ Useful for extracing bouts from velocity or vigor
    :param vel:
    :param threshold:
    :param min_duration:
    :param pad_before:
    :param pad_after:
    :return:
    """
    bouts = []
    in_bout = False
    start = 0
    connected = []
    continuity = False
    i = pad_before + 1
    bout_ended = pad_before
    while i < vel.shape[0] - pad_after:
        if np.isnan(vel[i]):
            continuity = False
            if in_bout and skip_nan:
                in_bout = False

        elif i > bout_ended and vel[i - 1] < threshold < vel[i] and not in_bout:
            in_bout = True
            start = i - pad_before

        elif vel[i - 1] > threshold > vel[i] and in_bout:
            in_bout = False
            if i - start > min_duration:
                bouts.append((start, i + pad_after))
                bout_ended = i + pad_after
                if continuity:
                    connected.append(True)
                else:
                    connected.append(False)
            continuity = True

        i += 1

    return bouts, connected