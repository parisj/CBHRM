from collections import deque
from typing import List, Tuple
import numpy as np


class Blackboard:
    def __init__(self, max_len: int) -> None:
        # Frame and ROI
        self.frame = None
        self.roi = None

        self.max_len = max_len

        # RGB signals
        self.samples_r = deque(maxlen=max_len)
        self.samples_b = deque(maxlen=max_len)
        self.samples_g = deque(maxlen=max_len)

        # Head signals
        self.samples_head_x = deque(maxlen=max_len)
        self.samples_head_y = deque(maxlen=max_len)
        self.samples_head_z = deque(maxlen=max_len)

        # resampled signals
        self.resample_rgb = None
        self.resample_head = None

        self.samples_rPPG = None
        self.samples_rhythmic = None
        self.bandpass_filtered = None
        self.signal_processor = None
        self.time_stamp = deque(
            maxlen=max_len,
        )
        self.array_rPPGs = np.empty((0, 256), dtype=np.float32)
        self.post_processed_rPPG = None
        self.weights = None
        self.hr = deque(maxlen=2)

    def update_frame(self, frame: np.array) -> None:
        self.frame = frame

    def update_roi(self, roi: np.array) -> None:
        self.roi = roi

    def update_samples_rgb(self, sample_rgb) -> None:
        self.samples_r.append(sample_rgb[2])
        self.samples_g.append(sample_rgb[1])
        self.samples_b.append(sample_rgb[0])

    def update_samples_head_pose(self, head_pose: np.array) -> None:
        self.samples_head_x.append(head_pose[0])
        self.samples_head_y.append(head_pose[1])
        self.samples_head_z.append(head_pose[2])

    def update_resampled(self, resample_rgb, resample_head) -> None:
        self.resample_rgb = resample_rgb
        self.resample_head = resample_head

    def update_time_stamp(self, time_stamp: float) -> None:
        self.time_stamp.append(time_stamp)

    def update_resampled(self, resample_rgb, resample_head) -> None:
        self.resample_rgb = resample_rgb
        self.resample_head = resample_head

    def update_samples_rPPG(self, rPPG: np.array) -> None:
        self.samples_rPPG = rPPG

    def update_samples_rhythmic(self, rhythmic: np.array) -> None:
        self.samples_rhythmic = rhythmic

    def update_bandpass_filtered(self, bandpass_filtered: np.array) -> None:
        self.bandpass_filtered = bandpass_filtered
        self.array_rPPGs = np.vstack((self.array_rPPGs, bandpass_filtered))

    def update_post_processed_rPPG(self, post_processed_rPPG: np.array) -> None:
        self.post_processed_rPPG = post_processed_rPPG

    def update_weights(self, weights: np.array) -> None:
        self.weights = weights

    def update_hr(self, hr: float) -> None:
        self.hr.append(hr)

    def get_frame(self) -> np.array:
        return self.frame

    def get_roi(self) -> np.array:
        return self.roi

    def get_samples_rgb(self) -> deque:
        return self.samples_r, self.samples_g, self.samples_b

    def get_samples_head_pose(self) -> deque:
        return self.samples_head_x, self.samples_head_y, self.samples_head_z

    def get_resamples_rgb(self) -> np.array:
        return self.resample_rgb

    def get_resamples_head(self) -> np.array:
        return self.resample_head

    def get_samples_rPPG(self) -> deque:
        return self.samples_rPPG

    def get_samples_rhythmic(self) -> deque:
        return self.samples_rhythmic

    def get_time_stamp(self) -> deque:
        return self.time_stamp

    # TODO Expand this method to return all samples
    def get_samples(self) -> Tuple:
        samples_rgb = self.get_samples_rgb()
        samples_head_pose = self.get_samples_head_pose()
        return self.frame, self.roi, samples_rgb, samples_head_pose

    def get_samples_signals(self) -> Tuple:
        samples_rgb = self.get_samples_rgb()
        samples_head_pose = self.get_samples_head_pose()
        time_stamps = self.get_time_stamp()
        return samples_rgb, samples_head_pose, time_stamps

    def get_bandpass_filtered(self) -> np.array:
        return self.bandpass_filtered

    def get_array_rPPGs(self) -> np.array:
        return self.array_rPPGs

    def get_post_processed_rPPG(self) -> np.array:
        return self.post_processed_rPPG

    def get_weights(self) -> np.array:
        return self.weights

    def get_hr(self) -> float:
        return self.hr
