from collections import deque
from typing import List, Tuple
import numpy as np
from threading import Lock
from scipy.signal import filtfilt, butter
import toml


class Blackboard:
    def __init__(self, max_len: int, initial_samples_event) -> None:
        # Frame and ROI
        self.settings = toml.load("settings.toml")
        self.settings_filtering = self.settings["filtering"]
        self.settings_evaluation = self.settings["evaluation"]
        self.len_hr_min = self.settings_evaluation["len_hr_min"]
        self.lowpass_order = self.settings_filtering["lowpass_order"]
        self.wn_lowpass = self.settings_filtering["wn_lowpass"]
        self.fs = self.settings_filtering["fs"]
        self.frame = None
        self.roi = None
        self.initial_samples_event = initial_samples_event
        self.max_len = max_len
        self.count_rPPG = 0
        self.b, self.a = butter(
            N=self.lowpass_order, Wn=self.wn_lowpass, btype="low", fs=self.fs
        )
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
        self.small_rPPGs = np.empty((0, 256), dtype=np.float32)
        self.post_processed_rPPG = None
        self.weights = None
        self.hr = deque(maxlen=11230)
        self.hr_plot = None
        self.hrv_plot = None
        self.BOOL_REFERENCE = False
        self.hr_reference = None
        self.hrv_ref = None

        self.monitoring_data = None
        self.offset_monitoring_data = None
        self.peaks = None
        self.diff_peaks = None
        self.rmssd = deque(maxlen=11230)

        self.lf = None
        self.hf = None

    def update_frame(self, frame: np.array) -> None:
        self.frame = frame

    def update_roi(self, roi: np.array) -> None:
        self.roi = roi

    def update_samples_rgb(self, sample_rgb) -> None:
        self.samples_r.append(sample_rgb[2])
        self.samples_g.append(sample_rgb[1])
        self.samples_b.append(sample_rgb[0])

        if not self.initial_samples_event.is_set():
            if len(self.samples_r) == self.max_len:
                self.initial_samples_event.set()

    def update_samples_head_pose(self, head_pose: np.array) -> None:
        self.samples_head_x.append(head_pose[0])
        self.samples_head_y.append(head_pose[1])
        self.samples_head_z.append(head_pose[2])

    def update_resampled(self, resample_rgb: np.array, resample_head: np.array) -> None:
        self.resample_rgb = resample_rgb
        self.resample_head = resample_head

    def update_time_stamp(self, time_stamp: float) -> None:
        self.time_stamp.append(time_stamp)

    def update_samples_rPPG(self, rPPG: np.array) -> None:
        self.samples_rPPG = rPPG

    def update_samples_rhythmic(self, rhythmic: np.array) -> None:
        self.samples_rhythmic = rhythmic

    def update_bandpass_filtered(self, bandpass_filtered: np.array) -> None:
        self.small_rPPG = bandpass_filtered
        self.bandpass_filtered = bandpass_filtered
        self.count_rPPG += 1

    def update_post_processed_rPPG(self, post_processed_rPPG: np.array) -> None:
        self.post_processed_rPPG = post_processed_rPPG

    def update_weights(self, weights: np.array) -> None:
        self.weights = weights

    def update_hr(self, hr: float) -> None:
        self.hr.append(hr)
        self.hr_plot = np.array(self.hr)
        if self.hr_plot.size > self.len_hr_min:
            self.hr_plot = filtfilt(self.b, self.a, self.hr_plot)

    def update_bool_reference(self, bool_reference: bool) -> None:
        self.BOOL_REFERENCE = bool_reference

    def update_hr_reference(self, hr_reference: float) -> None:
        self.hr_reference = hr_reference

    def update_hrv_reference(self, hrv_reference: float) -> None:
        self.hrv_ref = hrv_reference

    def update_monitoring_data(self, monitoring_data: dict) -> None:
        self.monitoring_data = monitoring_data
        self.offset_monitoring_data = len(self.hr) - len(monitoring_data)

    def update_peaks(self, peaks: List) -> None:
        self.peaks = peaks

    def update_diff_peaks(self, diff_peaks: List) -> None:
        self.diff_peaks = diff_peaks

    def increment_count_rPPG(self, increment: int) -> None:
        self.count_rPPG += increment

    def update_rmssd(self, rmssd: float) -> None:
        self.rmssd.append(rmssd)
        self.hrv_plot = np.array(self.rmssd)
        if self.hrv_plot.size > self.len_hr_min:
            self.hrv_plot = filtfilt(self.b, self.a, self.hrv_plot)

    # self.rmssd = filtfilt(self.b, self.a, self.rmssd)

    def update_lf(self, lf: float) -> None:
        self.lf = lf

    def update_hf(self, hf: float) -> None:
        self.hf = hf

    def get_rmssd(self) -> float:
        return self.rmssd

    def get_lf(self) -> float:
        return self.lf

    def get_hf(self) -> float:
        return self.hf

    def get_peaks(self) -> List:
        return self.peaks

    def get_diff_peaks(self) -> List:
        return self.diff_peaks

    def get_monitoring_data(self) -> dict:
        return self.monitoring_data, self.offset_monitoring_data

    def get_hr_reference(self) -> float:
        return self.hr_reference

    def get_hrv_reference(self) -> float:
        return self.hrv_ref

    def get_bool_reference(self) -> bool:
        return self.BOOL_REFERENCE

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

    def get_small_rPPG(self) -> np.array:
        return self.small_rPPG

    def get_count_rPPG(self) -> int:
        return self.count_rPPG

    def get_post_processed_rPPG(self) -> np.array:
        return self.post_processed_rPPG

    def get_weights(self) -> np.array:
        return self.weights

    def get_hr(self) -> float:
        return self.hr

    def get_hr_plot(self) -> np.array:
        return self.hr_plot

    def get_hrv_plot(self) -> np.array:
        return self.hrv_plot

    def reset_attributes(self):
        self.monitoring_data = None
        self.offset_monitoring_data = None
