import numpy as np
from collections import deque
import cv2
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import (
    butter,
    sosfiltfilt,
    filtfilt,
    spectrogram,
    find_peaks,
    lfilter,
    iirfilter,
    welch,
)
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

"""
TODO: 
- [ ] implement signal processing
   | - [ ] Resampling, Interpolation
- [x] implement POS algorithm
- [x] implement 3D Head Pose Estimation
   | - [x] Pitch 
   | - [x] Roll
   | - [x] yaw
- [ ] Postprocessing
"""


class Signal_processor:
    def __init__(self, write_event) -> None:
        self.control_obj = None
        self.samples_rgb = None
        self.sample_head_pose = None
        self.sample_pos = None
        self.pose_head = None
        self.write_event = write_event

    def attach(self, control_obj: "control.Control") -> None:
        self.control_obj = control_obj

    def average_intensities_rgb(self, image: np.array) -> np.array:
        self.sample_rgb.average(image, axis=(0, 1))

    def update(self, image: np.array, landmark) -> None:
        self.calc_samples_rgb(image)
        self.calc_samples_head_pose(image, landmark)
        self.control.update()

    def rPPG_method(self, resampled_rgb):
        # prepare data
        r, g, b = resampled_rgb
        r = r.reshape(-1, 1)
        g = g.reshape(-1, 1)
        b = b.reshape(-1, 1)

        # samples shape = (N, 3), N = number of samples
        samples = np.hstack((r, g, b))

        # Projection matrix
        S_m = np.array([[0, 1, -1], [-2, 1, 1]])

        N = len(r)
        l = 32
        H = np.zeros(N)

        # Moving window
        # m=n-l+1>=0 ---> n range from 0 to N-l
        # with slicing: [n : n + l, :]
        for n in range(0, N - l):
            # pick samples from n to n+l
            # C shape = (3, l)
            C = np.array([samples[n : n + l, :]]).reshape(-1, 3).T

            # mean of each channel
            mean = np.mean(C, axis=1)
            diag_mean = np.diag(mean)
            C_n = np.matmul(np.linalg.inv(diag_mean), C)

            # projection
            S = np.matmul(S_m, C_n)

            # scaling
            scale = np.std(S[0, :]) / np.std(S[1, :])

            # POS calculation -> H = rPPG signal
            h = S[0, :] + scale * S[1, :]
            H[n : n + l] = H[n : n + l] + (h - np.mean(h))

        self.control_obj.blackboard.update_samples_rPPG(H)

    def rhythmic_noise_surpression(self, resampled_head):
        sampling_rate = 20
        rPPG = self.control_obj.get_samples_rPPG()
        head_x, head_y, head_z = resampled_head

        # FFT of signals
        rPPG_fft = fft(rPPG)
        head_x_fft, head_y_fft, head_z_fft = fft(head_x), fft(head_y), fft(head_z)

        # Average and normalise head movements

        head_rhythmic = (
            self.norm_amp(head_x_fft)
            + self.norm_amp(head_y_fft)
            + self.norm_amp(head_z_fft)
        ) / 3
        head_rhythmic_norm = head_rhythmic / np.abs(head_rhythmic).max()

        # Remove rhythmic noise
        rhythmic_noise_suppressed = self.norm_amp(rPPG_fft) - head_rhythmic_norm

        # Mask frequencies outside the range [0.7, 4] Hz
        freqs = fftfreq(len(rPPG), 1 / sampling_rate)
        rhythmic_noise_suppressed[(freqs < 0.7) | (freqs > 4)] = 0

        # IFFT to get back to time domain
        filtered_rPPG = np.real(ifft(rhythmic_noise_suppressed))

        # Update blackboard
        self.control_obj.blackboard.update_samples_rhythmic(filtered_rPPG)

        # Frequency-domain bandpass filtering
        valid_freqs = (freqs >= 0.7) & (freqs <= 4)

        # Center frequency of the bandpass filter
        highest_freq = freqs[valid_freqs][
            np.argmax(np.abs(rhythmic_noise_suppressed[valid_freqs]))
        ]

        # Upper and lower cutoff frequencies
        high = highest_freq + 0.235
        low = highest_freq - 0.235

        # Design 50th-order Butterworth bandpass filter
        # sos = iirfilter(
        #    25,
        #    Wn=[low, high],
        #    fs=sampling_rate,
        #    btype="bandpass",
        #    ftype="butter",
        #    output="sos",
        # )
        sos = butter(
            N=35, Wn=[low, high], btype="bandpass", fs=sampling_rate, output="sos"
        )

        # Apply filter
        bandpass_filtered = sosfiltfilt(sos, rPPG, padlen=255)
        bandpass_filtered = (bandpass_filtered - np.mean(bandpass_filtered)) / np.std(
            bandpass_filtered
        )
        # Update blackboard
        self.control_obj.blackboard.update_bandpass_filtered(bandpass_filtered)

    def signal_processing_function(
        self, stop_event, initial_samples_event, new_sample_event
    ) -> None:
        fps = 20
        while not stop_event.is_set():
            initial_samples_event.wait()
            # if self.control_obj.blackboard.get_frame() is None:
            #    time.sleep(0.1)
            #    continue

            # if len(self.control_obj.blackboard.get_samples_rgb()[0]) <= 255:
            #    time.sleep(0.1)
            #    continue
            new_sample_event.wait()
            start_time = time.time()
            # Processing steps
            (
                samples_rgb,
                samples_head,
                time_stamps,
            ) = self.control_obj.blackboard.get_samples_signals()

            resampled_rgb, resampled_head = self.resample_samples(
                samples_rgb, samples_head, time_stamps
            )

            self.rPPG_method(resampled_rgb)
            self.rhythmic_noise_surpression(resampled_head)
            self.post_process_rPPG(fps)

            # Time managment
            duration = time.time() - start_time
            # sleep_time = max(0, 1 / 30 - duration)
            # time.sleep(sleep_time)
            if duration > 1 / 20:
                print("time to process longer than 30 fps", duration)
            new_sample_event.clear()

    def resample_to_target_time(self, samples, time_stamps, target_framerate=20):
        total_time = sum(time_stamps)
        target_time_intervals = np.linspace(0, total_time, 256)

        # Create a function for linear interpolation using SciPy
        interp_func = interp1d(
            np.cumsum(time_stamps), samples, kind="linear", fill_value="extrapolate"
        )

        # Interpolate the samples at the desired time intervals
        interpolated_values = interp_func(target_time_intervals)

        return interpolated_values

    def resample_samples(self, samples_rgb, samples_head, time_stamps):
        # Resample the RGB samples
        resampled_rgb = []
        resampled_head = []

        for sample in samples_rgb:
            rescolor = self.resample_to_target_time(sample, time_stamps)
            resampled_rgb.append(rescolor)

        # Resample the head pose samples
        for sample in samples_head:
            reshead = self.resample_to_target_time(sample, time_stamps)
            resampled_head.append(reshead)

        # Update the blackboard
        self.control_obj.blackboard.update_resampled(resampled_rgb, resampled_head)
        return resampled_rgb, resampled_head

    def norm_amp(self, signal) -> np.array:
        return signal / np.abs(signal).max()

    def post_process_rPPG(self, fps) -> None:
        array_rPPGs = self.control_obj.blackboard.get_array_rPPGs()
        old_weights = self.control_obj.blackboard.get_weights()
        final_rPPG = self.control_obj.blackboard.get_post_processed_rPPG()
        if final_rPPG is not None:
            final_rPPG = final_rPPG.copy()
        f_resample = fps
        # depending on how often the rPPG signal is caluclated
        # if it is calculated every sample it's 30
        f_recalculated = fps

        # Length of the rPPG signals
        signal_length = array_rPPGs.shape[1]

        # Calculate the total shift that will occur
        number_signals = array_rPPGs.shape[0]
        shift_amount = int(f_resample / f_recalculated)
        # Calculate the total length of the final_rPPG
        final_length = int(signal_length + (number_signals - 1) * shift_amount)
        weights = np.zeros(final_length)

        if old_weights is None and final_rPPG is None:
            final_rPPG = array_rPPGs[0]
            weights = np.ones(final_length)

        elif number_signals > 0 and number_signals <= signal_length:
            old_weights = self.control_obj.blackboard.get_weights()
            flanks = range(1, min(number_signals, signal_length))
            weights[0 : max(flanks)] = flanks
            weights[-max(flanks) :] = flanks[::-1]
            weights[weights == 0] = min(number_signals, 256)
            non_scaled_rPPG = final_rPPG * old_weights
            final_rPPG = np.append(non_scaled_rPPG, np.zeros(shift_amount))
            final_rPPG[-256:] += array_rPPGs[-1]
            final_rPPG = final_rPPG / weights

        elif number_signals > signal_length:
            old_weights = np.arange(255, 0, -1)
            weights = np.arange(256, 0, -1)
            final_rPPG[-255:] = final_rPPG[-255:] * old_weights
            new_int = array_rPPGs[-1]
            final_rPPG = np.append(final_rPPG, np.zeros(shift_amount))
            final_rPPG[-256:] += new_int
            final_rPPG[-256:] = final_rPPG[-256:] / weights

        self.control_obj.blackboard.update_weights(weights)
        self.control_obj.blackboard.update_post_processed_rPPG(final_rPPG)
        self.heart_rate()

    def heart_rate(self) -> None:
        time_window = 300
        signal = self.control_obj.blackboard.get_post_processed_rPPG()
        peaks, _ = find_peaks(signal[-time_window:-10], distance=7, prominence=0.1)
        diff_peaks = np.diff(peaks)
        IBI = np.mean(diff_peaks) / 20
        HR = (1 / IBI) * 60
        rmssd = self.calculate_RMSSD(peaks)
        # lf, hf = self.calculate_LF_HF(peaks)
        self.control_obj.blackboard.update_hr(HR)
        self.control_obj.blackboard.update_peaks(peaks)
        self.control_obj.blackboard.update_diff_peaks(diff_peaks)
        self.control_obj.blackboard.update_rmssd(rmssd)
        # self.control_obj.blackboard.update_lf(lf)
        # self.control_obj.blackboard.update_hf(hf)
        self.write_event.set()

    def calculate_RMSSD(self, peaks):
        """
        Calculate the RMSSD (Root Mean Square of Successive Differences) for HRV

        Returns:
        float: The RMSSD value.
        """

        # Convert indices to time intervals (in seconds)
        time_intervals = np.array(peaks) / 20

        # Calculate successive differences
        diff = np.diff(time_intervals)

        # Calculate RMSSD
        rmssd = np.sqrt(np.mean(diff**2))
        return rmssd

    def calculate_LF_HF(self, peaks):
        """
        Calculate the LF (Low Frequency) and HF (High Frequency) components of HRV.

        Returns:
        tuple: The LF and HF values (normalized with the total power of hf and lf).
        """
        # Convert indices to time intervals (in seconds)

        time_intervals = np.array(peaks) / 20

        # Interpolate IBIs at 2.5Hz
        interpolated_IBIs = np.interp(
            np.arange(0, time_intervals[-1], 0.4), time_intervals, time_intervals
        )

        # Calculate power spectral density using Welch's method
        f, Pxx = welch(interpolated_IBIs, fs=2.5, nperseg=len(interpolated_IBIs))

        # Define LF and HF bands
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)

        # Calculate power in LF and HF bands
        lf_power = np.trapz(
            Pxx[(f >= lf_band[0]) & (f <= lf_band[1])],
            f[(f >= lf_band[0]) & (f <= lf_band[1])],
        )
        hf_power = np.trapz(
            Pxx[(f >= hf_band[0]) & (f <= hf_band[1])],
            f[(f >= hf_band[0]) & (f <= hf_band[1])],
        )

        # Normalize LF and HF
        total_power = lf_power + hf_power
        lf_normalized = lf_power / total_power
        hf_normalized = hf_power / total_power
        return lf_normalized, hf_normalized
