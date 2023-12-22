import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, sosfiltfilt
from scipy.fft import fft, fftfreq


def generate_sine_wave(freq, sample_rate, duration):
    """
    Generate a sine wave based on a given frequency, sample rate, and duration.

    :param freq: Frequency of the sine wave in Hz.
    :param sample_rate: Number of samples per second.
    :param duration: Duration of the wave in seconds.
    :return: Array representing the sine wave.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def butterworth_filter(data, cutoff, fs, order, btype):
    """
    Apply a Butterworth filter to a dataset.

    :param data: Input data to be filtered.
    :param cutoff: Cutoff frequency of the filter in Hz. For bandpass filters, this should be a tuple of (low, high).
    :param fs: Sampling rate in Hz.
    :param order: Order of the filter.
    :param btype: Type of filter (e.g., 'low', 'high', 'bandpass').
    :return: Filtered data.
    """
    nyq = 0.5 * fs
    normal_cutoff = [c / nyq for c in cutoff]  # Normalize each cutoff frequency
    sos = butter(order, normal_cutoff, btype=btype, output="sos")
    y = sosfiltfilt(sos, data)
    return y


def plot_frequency_domain(signal, sample_rate, title):
    """
    Plot the frequency domain representation of a signal.
    """
    N = len(signal)
    yf = np.fft.rfft(signal)
    xf = np.fft.rfftfreq(N, 1 / sample_rate)

    plt.plot(xf, np.abs(yf))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()


# Parameters
freq = np.random.uniform(0.7, 4)  # Random frequency between 0.7 and 4 Hz
print(freq)
sample_rate = 30  # Sample rate in Hz
duration = 16  # Duration in seconds
filter_order = 50
passband = 0.47  # Passband for the filter

# Generate original sine wave
sine_wave = generate_sine_wave(freq, sample_rate, duration)

# Add noise and other sine waves
noise = np.random.normal(0, 3, sine_wave.shape)
print(noise)
other_sines = (
    generate_sine_wave(8, sample_rate, duration)
    + generate_sine_wave(14, sample_rate, duration)
    + 0.3 * generate_sine_wave(1, sample_rate, duration)
    + 0.4 * generate_sine_wave(3, sample_rate, duration)
    + 0.2 * generate_sine_wave(6, sample_rate, duration)
    + 0.3 * generate_sine_wave(20, sample_rate, duration)
    + 0.3 * generate_sine_wave(3.2, sample_rate, duration)
)

signal = sine_wave + noise + other_sines

# Apply Butterworth filter
filtered_signal = butterworth_filter(
    signal,
    [freq - passband / 2, freq + passband / 2],
    sample_rate,
    filter_order,
    "bandpass",
)

# Plotting
plt.figure(figsize=(12, 6))

# Original Signal
plt.subplot(2, 1, 1)
plot_frequency_domain(signal, sample_rate, "Frequency Domain - Original Signal")
plt.axvline(x=(freq - passband / 2), color="r", linestyle="--", label="Cutoff")
plt.axvline(
    x=(freq + passband / 2),
    color="r",
    linestyle="--",
)
plt.title("Noisy Signal")
plt.xlim(0, 6)
plt.legend()

# Filtered Signal
plt.subplot(2, 1, 2)
plot_frequency_domain(
    filtered_signal, sample_rate, "Frequency Domain - Filtered Signal"
)
plt.axvline(x=(freq - passband / 2), color="r", linestyle="--", label="Cutoff")
plt.axvline(
    x=(freq + passband / 2),
    color="r",
    linestyle="--",
)
plt.title("Signal after 50th Order Butterworth Filter ")
plt.xlim(0, 6)
plt.legend()

plt.tight_layout()
plt.show()
