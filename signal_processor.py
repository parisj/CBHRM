import numpy as np
from collections import deque
import cv2
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import butter, sosfiltfilt, filtfilt, spectrogram, find_peaks
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
    def __init__(self) -> None:
        self.control_obj = None
        self.samples_rgb = None
        self.sample_head_pose = None
        self.sample_pos = None
        self.pose_head = None

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
        sampling_rate = 30
        rPPG = self.control_obj.get_samples_rPPG()
        head_x, head_y, head_z = resampled_head

        #       f, t, Sxx = spectrogram(
        #           rPPG, fs=sampling_rate, nperseg=64, noverlap=32, window="hann"
        #       )
        #       # Plotting
        #       plt.figure(figsize=(15, 8))
        #       plt.pcolormesh(t, f, Sxx, shading="gouraud")
        #       plt.ylabel("Frequency [Hz]")
        #       plt.xlabel("Time [sec]")
        #       plt.ylim([0, 50])
        #       plt.colorbar(label="Intensity")
        #       plt.show()

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

        #        f, t, Sxx = spectrogram(
        #            filtered_rPPG, fs=sampling_rate, nperseg=64, noverlap=32, window="hann"
        #        )
        #
        #        # Plotting
        #        plt.figure(figsize=(15, 8))
        #        plt.pcolormesh(t, f, Sxx, shading="gouraud")
        #        plt.ylabel("Frequency [Hz]")
        #        plt.xlabel("Time [sec]")
        #        plt.ylim([0, 50])
        #        plt.colorbar(label="Intensity")
        #        plt.show()

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
        sos = butter(
            N=30, Wn=[low, high], btype="bandpass", fs=sampling_rate, output="sos"
        )

        # Apply filter
        bandpass_filtered = sosfiltfilt(sos, rPPG, padlen=255)
        bandpass_filtered = (bandpass_filtered - np.mean(bandpass_filtered)) / np.std(
            bandpass_filtered
        )
        # Update blackboard
        self.control_obj.blackboard.update_bandpass_filtered(bandpass_filtered)

    def signal_processing_function(self) -> None:
        while True:
            start_time = time.time()
            if self.control_obj.blackboard.get_frame() is None:
                time.sleep(1)
                continue

            if len(self.control_obj.blackboard.get_samples_rgb()[0]) <= 255:
                time.sleep(1)
                continue

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
            self.post_process_rPPG()

            # Time managment
            duration = time.time() - start_time
            sleep_time = max(0, 1 / 4 - duration)
            time.sleep(sleep_time)
            print(
                "time to process [rPPG, Rhythmic noise surpression, post processing]: ",
                duration,
            )
            # postprocessing steps
            # TODO HR Calculation
            # TODO HRV Calculation

    def resample_to_target_time(self, samples, time_stamps, target_framerate=30):
        total_time = sum(time_stamps)

        target_time_intervals = np.linspace(0, total_time, 256)

        # Create a function for linear interpolation using SciPy
        interp_func = interp1d(
            np.cumsum(time_stamps), samples, kind="linear", fill_value="extrapolate"
        )

        # Interpolate the samples at the desired time intervals
        interpolated_values = interp_func(target_time_intervals)

        # Create a new deque with the interpolated samples and their corresponding times

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

    def post_process_rPPG(self) -> None:
        array_rPPGs = self.control_obj.blackboard.get_array_rPPGs()
        old_weights = self.control_obj.blackboard.get_weights()
        final_rPPG = self.control_obj.blackboard.get_post_processed_rPPG()
        if final_rPPG is not None:
            final_rPPG = final_rPPG.copy()
        f_resample = 30
        # depending on how often the rPPG signal is caluclated
        # if it is calculated every sample it's 30
        f_recalculated = 30

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
            print("first time")
            # print("final_rPPG", final_rPPG.shape, "\n")
            # print("new weights", weights, "\n")
            # print("shape of new weights", weights.shape, "\n")

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
            print("middle")
            # print("final_rPPG", final_rPPG.shape, "\n")
            # print("old weights", old_weights, "\n", "new weights", weights, "\n")
            # print(
            #    "shape of old weights",
            #    old_weights.shape,
            #    "\n",
            #    "shape of new weights",
            #    weights.shape,
            #    "\n",
            # )

        elif number_signals > signal_length:
            old_weights = np.arange(255, 0, -1)
            weights = np.arange(256, 0, -1)
            final_rPPG[-255:] = final_rPPG[-255:] * old_weights
            new_int = array_rPPGs[-1]
            final_rPPG = np.append(final_rPPG, np.zeros(shift_amount))
            final_rPPG[-256:] += new_int
            final_rPPG[-256:] = final_rPPG[-256:] / weights
            print("full")
            # print("final_rPPG", final_rPPG.shape, "\n")
            # print("old weights", old_weights, "\n", "new weights", weights, "\n")
            # print(
            #    "shape of old weights",
            #    old_weights.shape,
            #    "\n",
            #    "shape of new weights",
            #    weights.shape,
            #    "\n",
            # )

        self.control_obj.blackboard.update_weights(weights)
        self.control_obj.blackboard.update_post_processed_rPPG(final_rPPG)
        self.heart_rate()

    def heart_rate(self) -> None:
        time_window = 450
        signal = self.control_obj.blackboard.get_post_processed_rPPG()
        peaks, _ = find_peaks(signal[-time_window:])
        IBI = np.mean(np.diff(peaks)) / 30
        HR = 1 / IBI
        self.control_obj.blackboard.update_hr(HR)
        print("HR: ", HR, "Scaled HR: ", HR * 60)
