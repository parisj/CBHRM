import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dash import dcc, html, Input, Output
from dash.dependencies import Input, Output
import dash
from dash.exceptions import PreventUpdate

# Info for FPS
import time


def real_time_face_landmark_detection():
    """
    Perform real-time face and face landmark detection.
    Shows the FPS on the rendered video.
    """
    fps = 30
    maxlen_samples = 256
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)

    prev_time = 0  # Previous frame time (For FPS)
    samples = deque(maxlen=maxlen_samples)
    # TODO import different model from mediapipe for cleaner landmarks
    # (https://google.github.io/mediapipe/solutions/face_mesh.html)
    # https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python
    # https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    plt.ion()
    fix, axs = plt.subplots(3, 1)
    (line_r,) = axs[0].plot([], [], "r", label="Red")
    (line_g,) = axs[1].plot([], [], "g", label="Green")
    (line_b,) = axs[2].plot([], [], "b", label="Blue")

    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.8, min_tracking_confidence=0.9
    ) as face_mesh:
        while cap.isOpened():
            ret, image = cap.read()

            if ret == False:
                break
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # process frame with face_mesh
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    landmark_list = list(face.landmark)
                    mask, cutout = create_mask(landmark_list, image.shape)
                    roi = extract_polygonal_roi(image, mask, cutout)
                cv2.imshow("Roi", roi)
                update_samples_and_plot(samples, roi, line_r, line_g, line_b, axs)
            # Display FPS
            cv2.putText(
                image,
                f"FPS: {int(fps)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("MediaPipe Face Mesh", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        plt.ioff()
        cap.release()
        cv2.destroyAllWindows()


def create_mask(landmark, image_shape):
    facemask_points = [151, 299, 298, 411, 410, 11, 186, 187, 68, 69]
    left_eye_points = [
        226,
        113,
        225,
        224,
        223,
        222,
        221,
        189,
        245,
        233,
        232,
        231,
        230,
        229,
        228,
        31,
        226,
    ]
    right_eye_points = [
        446,
        342,
        445,
        445,
        444,
        443,
        442,
        441,
        413,
        465,
        453,
        452,
        451,
        450,
        449,
        448,
        446,
    ]
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    face = np.array(
        [
            [int(landmark[i].x * image_shape[1]), int(landmark[i].y * image_shape[0])]
            for i in facemask_points
        ]
    )
    left_eye = np.array(
        [
            [int(landmark[i].x * image_shape[1]), int(landmark[i].y * image_shape[0])]
            for i in left_eye_points
        ]
    )
    right_eye = np.array(
        [
            [int(landmark[i].x * image_shape[1]), int(landmark[i].y * image_shape[0])]
            for i in right_eye_points
        ]
    )

    cv2.fillPoly(mask, [face], 1)
    cv2.fillPoly(mask, [left_eye], 0)
    cv2.fillPoly(mask, [right_eye], 0)
    min_x = np.min(face[:, 0])
    max_x = np.max(face[:, 0])
    min_y = np.min(face[:, 1])
    max_y = np.max(face[:, 1])

    cutout = [min_x, max_x, min_y, max_y]

    return mask, cutout


def average_intensities(image):
    return np.average(image, axis=(0, 1))


def extract_polygonal_roi(image, mask, cutout):
    """
    Extract the polygonal ROI from the image based on the given mask.

    Parameters:
    image (np.array): The source image.
    mask (np.array): The boolean or uint8 mask.

    Returns:
    np.array: The extracted ROI.
    """
    min_x, max_x, min_y, max_y = cutout
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask_expanded = np.expand_dims(mask, axis=2)

    roi = (
        image[min_y:max_y, min_x:max_x, :] * mask_expanded[min_y:max_y, min_x:max_x, :]
    )

    return roi


def update_samples_and_plot(samples, image, line_r, line_g, line_b, axs):
    """
    Update the samples deque and plot the new samples.

    Parameters:
    samples (deque): The samples deque to update.
    image (np.array): The source image from which to extract the samples.
    line_r, line_g, line_b (Line2D): Matplotlib Line2D objects for the plot.
    axs (array): Array of subplot axes.
    """
    sample = average_intensities(image)
    samples.append(sample)

    if len(samples) > 1:
        x = np.arange(len(samples))
        line_r.set_data(x, [s[0] for s in samples])
        line_g.set_data(x, [s[1] for s in samples])
        line_b.set_data(x, [s[2] for s in samples])

        for ax in axs:
            ax.relim()
            ax.autoscale_view()

        plt.pause(0.01)


if __name__ == "__main__":
    real_time_face_landmark_detection()
