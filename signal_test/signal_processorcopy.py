import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import (
    butter,
    sosfiltfilt,
    find_peaks,
    welch,
)
import toml
import time
from scipy.interpolate import interp1d

# from viztracer import VizTracer
from src.util.livefilter import RealTimeFilter


class Signal_processor:
    def __init__(self, write_event) -> None:
        self.control_obj = None
        self.samples_rgb = None
        self.sample_head_pose = None
        self.sample_pos = None
        self.pose_head = None
        self.write_event = write_event
        self.b, self.a = butter(N=4, Wn=0.6, btype="low", fs=20)
        self.lfilter = RealTimeFilter(self.b, self.a)

        self.settings = toml.load("settings.toml")
        self.fps = self.settings["camera"]["fps_camera"]
        self.bandpass_order = self.settings["filtering"]["bandpass_order"]

    def attach(self, control_obj: "control.Control") -> None:
        self.control_obj = control_obj

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

            # mean of each channel, temporal normalisation
            mean = np.mean(C, axis=1)
            diag_mean = np.diag(mean)
            C_n = np.matmul(np.linalg.inv(diag_mean), C)

            # projection
            S = np.matmul(S_m, C_n)

            # scaling
            alpha = np.std(S[0, :]) / np.std(S[1, :])

            # POS calculation -> H = rPPG signal
            h = S[0, :] + alpha * S[1, :]
            H[n : n + l] = H[n : n + l] + (h - np.mean(h))

        self.control_obj.blackboard.update_samples_rPPG(H)

    def rhythmic_noise_surpression(self, resampled_head):
        sampling_rate = self.fps
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
        rhythmic_noise_suppressed = rPPG_fft - head_rhythmic_norm

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
        high = highest_freq + self.settings["filtering"]["cutoff_window"][0]
        low = highest_freq + self.settings["filtering"]["cutoff_window"][1]
        N = self.bandpass_order
        # Create filter coefficients
        sos = butter(
            N=N, Wn=[low, high], btype="bandpass", fs=sampling_rate, output="sos"
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
        fs = self.fps
        delay_tolerance = 0.8

        # tracer = VizTracer()
        # tracer.start()
        while not stop_event.is_set():
            initial_samples_event.wait()

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
            self.post_process_rPPG(start_time, fs, delay_tolerance)

            # Time managment
            duration = time.time() - start_time
            if duration > 1 / fs:
                print(
                    "WARNING: PROCESSING LONGER THAN SAMPLING FREQUENCY ALLOWS",
                    duration,
                )
            new_sample_event.clear()
        # tracer.stop()
        # tracer.save(output_file="result.json")

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

    def post_process_rPPG(self, start_time, fs, delay_tolerance) -> None:
        delay = False
        shift_delay = 0
        small_rPPG = self.control_obj.blackboard.get_small_rPPG()
        number_signals = self.control_obj.blackboard.get_count_rPPG()
        old_weights = self.control_obj.blackboard.get_weights()
        final_rPPG = self.control_obj.blackboard.get_post_processed_rPPG()
        if final_rPPG is not None:
            final_rPPG = final_rPPG.copy()
        f_resample = fs
        # depending on how often the rPPG signal is caluclated
        # if it is calculated every sample it's fps
        f_recalculated = fs
        duration = time.time() - start_time
        if duration > 1 / fs:
            delay = True
            shift_delay = int(duration * fs)
            number_signals += shift_delay
            self.control_obj.blackboard.increment_count_rPPG(shift_delay)
        # Length of the rPPG signals
        signal_length = 256

        # Calculate the total shift that will occur

        shift_amount = int(f_resample / f_recalculated)
        # Calculate the total length of the final_rPPG
        final_length = int(signal_length + (number_signals - 1) * shift_amount)
        weights = np.zeros(final_length)
        if old_weights is None and final_rPPG is None:
            final_rPPG = small_rPPG
            weights = np.ones(final_length)

        elif number_signals > 0 and number_signals <= signal_length:
            old_weights = self.control_obj.blackboard.get_weights()
            flanks = range(1, min(number_signals, signal_length))
            weights[0 : max(flanks)] = flanks
            weights[-max(flanks) :] = flanks[::-1]
            weights[weights == 0] = min(number_signals, 256)
            non_scaled_rPPG = final_rPPG * old_weights
            final_rPPG = np.append(
                non_scaled_rPPG, np.zeros(shift_amount + shift_delay)
            )
            final_rPPG[-256:] += small_rPPG
            final_rPPG = final_rPPG / weights

        elif number_signals > signal_length:
            old_weights = np.arange(255, 0, -1)
            weights = np.arange(256, 0, -1)
            final_rPPG[-255 + shift_delay :] = (
                final_rPPG[-255 + shift_delay :] * old_weights[shift_delay:]
            )
            new_int = small_rPPG
            final_rPPG = np.append(final_rPPG, np.zeros(shift_amount + shift_delay))
            final_rPPG[-256:] += new_int
            final_rPPG[-256:] = final_rPPG[-256:] / weights

        self.control_obj.blackboard.update_weights(weights)
        self.control_obj.blackboard.update_post_processed_rPPG(final_rPPG)

        self.heart_rate(delay, shift_delay)

    def heart_rate(self, delay=False, shift=None) -> None:
        time_window = self.settings["heart_rate"]["time_window"]
        start_delay_peak_detection = self.settings["heart_rate"][
            "start_delay_peak_detection"
        ]

        signal = self.control_obj.blackboard.get_post_processed_rPPG()
        peaks, _ = find_peaks(
            signal[-time_window:-start_delay_peak_detection], distance=6, prominence=0.1
        )
        diff_peaks = np.diff(peaks)

        IBI = np.mean(diff_peaks) / self.fps
        HR = (1 / IBI) * 60
        # HR = self.lfilter.process_sample(HR)

        rmssd = self.calculate_RMSSD(peaks) * 1000
        # rmssd = self.lfilter.process_sample(rmssd)
        # lf, hf = self.calculate_LF_HF(peaks)
        self.control_obj.blackboard.update_hr(HR)
        self.control_obj.blackboard.update_peaks(peaks)
        self.control_obj.blackboard.update_diff_peaks(diff_peaks)
        self.control_obj.blackboard.update_rmssd(rmssd)
        # self.control_obj.blackboard.update_lf(lf)
        # self.control_obj.blackboard.update_hf(hf)
        if delay:
            for i in range(shift):
                self.control_obj.blackboard.update_hr(HR)
                self.control_obj.blackboard.update_peaks(peaks)
                self.control_obj.blackboard.update_diff_peaks(diff_peaks)
                self.control_obj.blackboard.update_rmssd(rmssd)
                self.write_event.set()
        self.write_event.set()

    def calculate_RMSSD(self, peaks):
        # Convert indices to time intervals (in seconds)
        time_intervals = np.array(peaks) / self.fps

        # Calculate successive differences
        diff = np.diff(time_intervals)

        # Calculate RMSSD
        rmssd = np.sqrt(np.mean(diff**2))
        return rmssd

    def calculate_LF_HF(self, peaks):
        # Convert indices to time intervals (in seconds)

        time_intervals = np.array(peaks) / self.fps

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
