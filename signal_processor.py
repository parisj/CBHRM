import numpy as np
from collections import deque
import cv2
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import butter, sosfiltfilt, filtfilt
import time
import matplotlib.pyplot as plt

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

    def resample_signal(signal, target_fps, total_frames):
        pass

    def rPPG_method(self):
        # prepare data
        r, g, b = self.control_obj.blackboard.get_samples_rgb()
        r = np.array(r, dtype=np.uint8).reshape(-1, 1)
        g = np.array(g, dtype=np.uint8).reshape(-1, 1)
        b = np.array(b, dtype=np.uint8).reshape(-1, 1)

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

    def rhythmic_noise_surpression(self):
        sampling_rate = 30
        rPPG = self.control_obj.get_samples_rPPG()
        head_x, head_y, head_z = self.control_obj.blackboard.get_samples_head_pose()

        # FFT of signals
        rPPG_fft = fft(rPPG)
        head_x_fft, head_y_fft, head_z_fft = fft(head_x), fft(head_y), fft(head_z)

        # Average and normalise head movements
        head_rhythmic = (head_x_fft + head_y_fft + head_z_fft) / 3
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
        high = highest_freq + 0.235
        low = highest_freq - 0.235

        # Design 50th-order Butterworth bandpass filter
        b, a = butter(N=50, Wn=[low, high], btype="bandpass", fs=sampling_rate)

        # Apply filter
        bandpass_filtered = filtfilt(b, a, rPPG, padlen=255)

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
            self.rPPG_method()
            self.rhythmic_noise_surpression()

            # Time managment
            duration = time.time() - start_time
            sleep_time = max(0, 1 / 4 - duration)
            time.sleep(sleep_time)

            # postprocessing steps
            # TODO HR Calculation
            # TODO HRV Calculation
