import cv2
import toml
import mediapipe as mp
import time
import numpy as np
import control
from video_stream import VideoStream


def create_mask(lm, image_shape: np.array, mask_lm: dict) -> (np.array, list):
    # Extract the face mask points from the landmark list
    facemask_p = mask_lm["facemask_points"]
    left_eye_p = mask_lm["left_eye_points"]
    right_eye_p = mask_lm["right_eye_points"]

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # Create the mask from the landmark points with their location in the image
    face = np.array(
        [
            [int(lm[i].x * image_shape[1]), int(lm[i].y * image_shape[0])]
            for i in facemask_p
        ]
    )
    left_eye = np.array(
        [
            [int(lm[i].x * image_shape[1]), int(lm[i].y * image_shape[0])]
            for i in left_eye_p
        ]
    )
    right_eye = np.array(
        [
            [int(lm[i].x * image_shape[1]), int(lm[i].y * image_shape[0])]
            for i in right_eye_p
        ]
    )

    # fill mask
    cv2.fillPoly(mask, [face], 1)
    cv2.fillPoly(mask, [left_eye], 0)
    cv2.fillPoly(mask, [right_eye], 0)

    # Get the roi coordinates
    min_x = np.min(face[:, 0])
    max_x = np.max(face[:, 0])
    min_y = np.min(face[:, 1])
    max_y = np.max(face[:, 1])

    cutout = [min_x, max_x, min_y, max_y]
    # print(cutout)
    return mask, cutout


def extract_polygonal_roi(frame: np.array, mask: np.array, cutout: list) -> np.array:
    # Extract the ROI coordinates
    min_x, max_x, min_y, max_y = cutout

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Expand the mask to 3 channels
    mask_expanded = np.expand_dims(mask[min_y : max_y + 1, min_x : max_x + 1], axis=2)

    # Extract the ROI
    roi = frame[min_y : max_y + 1, min_x : max_x + 1].copy()

    # Apply the mask to the ROI
    roi = roi * mask_expanded
    # print(f"ROI shape: {roi.shape}, Mask Expanded shape: {mask_expanded.shape}")

    # print(roi.shape[0:2] == mask_expanded.shape[0:2])

    return roi


def mean_intensities_rgb(roi: np.array) -> np.array:
    return np.mean(roi, axis=(0, 1))


def calculate_head_pose(
    head_lm_2d: np.array, head_lm_3d: np.array, frame_shape
) -> list:
    # Set the camera matrix and distortion
    cam_matrix = np.array(
        [
            [1.0 * frame_shape[1], frame_shape[0] / 2, 0.0],
            [0.0, 1.0 * frame_shape[1], frame_shape[1] / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype="double",
    )
    camera_distortion = np.zeros((4, 1))

    # Calculate the head pose angles
    _, rotation_vec, translation_vec = cv2.solvePnP(
        head_lm_3d, head_lm_2d, cam_matrix, camera_distortion
    )
    rmat, _ = cv2.Rodrigues(rotation_vec)

    # Decompose the rotation matrix into euler angles
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 3600 * 2
    # print(angles)
    return (x, y, z)


def headpose_landmarks(
    landmarks: list, landmarks_dict: dict, image_shape: list
) -> list:
    # Get the head pose landmarks
    lm = landmarks
    lm_pose = landmarks_dict["head_pose_points"]

    # Create Image and World points for the head pose estimation
    head_lm_2d = np.array(
        [
            [int(lm[i].x * image_shape[1]), int(lm[i].y * image_shape[0])]
            for i in lm_pose
        ],
        dtype=np.float64,
    )

    head_lm_3d = np.array(
        [
            [
                int(lm[i].x * image_shape[1]),
                int(lm[i].y * image_shape[0]),
                lm[i].z,
            ]
            for i in lm_pose
        ],
        dtype=np.float64,
    )

    return head_lm_2d, head_lm_3d


def face_processing(
    control_pass, took_sample_event, stop_event, new_sample_event
) -> (np.array, np.array, list):
    # Set the shared control object
    control_obj = control_pass

    # Start the video stream (Thread start)
    video_stream = VideoStream(took_sample_event, stop_event).start()

    # Initialise the face mesh
    mp_face_mesh = mp.solutions.face_mesh

    # Initialise the previous time for FPS calculation
    prev_time = 0

    # load settings toml file
    settings = toml.load("settings.toml")

    # dictionary of landmark points for facemask and headpose
    landmark_dict = settings["facemask"]

    # start face mesh processing
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.8, min_tracking_confidence=0.9
    ) as face_mesh:
        # loop over frames from the video stream
        for frame in video_stream.update():
            # Calculate FPS
            current_time = time.time()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # process frame with face_mesh
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                # print("face detected \n")
                for face in results.multi_face_landmarks:
                    # print("face", "\n")
                    # extract landmark points from frame
                    landmark_list = list(face.landmark)

                    # create mask, roi and headpose
                    mask, cutout = create_mask(
                        landmark_list, frame.shape, landmark_dict
                    )

                    roi = extract_polygonal_roi(frame, mask, cutout)
                    head_lm_2d, head_lm_3d = headpose_landmarks(
                        landmark_list, landmark_dict, frame.shape
                    )

                    pose_angles = calculate_head_pose(
                        head_lm_2d, head_lm_3d, frame.shape
                    )

                    # print("pose_angles", pose_angles, "\n")
                    # retrieve rgb samples from roi
                    rgb_samples = mean_intensities_rgb(roi)
                    duration = current_time - prev_time
                    # sleep_time = max(0, 1 / 30 - duration)
                    # time.sleep(sleep_time)
                    control_obj.update_samples(
                        frame, roi, rgb_samples, pose_angles, duration
                    )
                    new_sample_event.set()
                    # cv2.imshow("Roi", roi)
                    # if fps < 28 or fps > 32:
                    #    print("FPS: ", fps, "\n")
                    #    print("time: ", current_time - prev_time, "\n")

                    prev_time = current_time
