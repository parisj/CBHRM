import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def remove_black_pixels(image_array):
    mask = image_array != 0
    return image_array[mask]


def plot_histogram(image_array, title, color, label):
    plt.hist(image_array.ravel(), bins=256, color=color, alpha=0.5, label=label)
    mean_intensity = np.mean(image_array)
    plt.axvline(
        mean_intensity,
        color=color,
        linestyle="dashed",
        linewidth=2,
        alpha=1,
        label=f"Mean Intensity: {mean_intensity:.2f}",
    )


roi_dir = "dataset/New folder/"

roi_cleaned_list = []

for i in range(1, 5):
    roi_path = os.path.join(roi_dir, f"p{i}_roi.png")
    roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    roi_clean = remove_black_pixels(roi)
    roi_cleaned_list.append(roi_clean)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

labels = ["P1 ROI", "P2 ROI", "P3 ROI", "P4 ROI"]

plt.figure(figsize=(10, 4))

for i, roi_clean in enumerate(roi_cleaned_list):
    title = f"Pixel Intensity Histogram for {labels[i]} (Gray)"
    plot_histogram(roi_clean, title, colors[i], labels[i])

plt.title("Pixel Intensity Histograms for ROIs")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.show()
