import cv2
import toml
import mediapipe as mp
import time
import numpy as np
import control


def capture_frame() -> np.array:
    WIDTH = 1920
    HEIGHT = 1080
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        print(frame.shape)
        frame = frame[250:800, 640:1280]
        cv2.imshow("frame", frame)
        yield frame
    cap.release()
    cv2.destroyAllWindows()
    return 0


def create_mask(lm, image_shape: np.array, mask_lm: dict) -> (np.array, list):
    facemask_p = mask_lm["facemask_points"]
    left_eye_p = mask_lm["left_eye_points"]
    right_eye_p = mask_lm["right_eye_points"]

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
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

    cv2.fillPoly(mask, [face], 1)
    cv2.fillPoly(mask, [left_eye], 0)
    cv2.fillPoly(mask, [right_eye], 0)
    min_x = np.min(face[:, 0])
    max_x = np.max(face[:, 0])
    min_y = np.min(face[:, 1])
    max_y = np.max(face[:, 1])

    cutout = [min_x, max_x, min_y, max_y]
    # print(cutout)
    return mask, cutout


def extract_polygonal_roi(frame: np.array, mask: np.array, cutout: list) -> np.array:
    min_x, max_x, min_y, max_y = cutout

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask_expanded = np.expand_dims(mask[min_y : max_y + 1, min_x : max_x + 1], axis=2)

    # Extract the ROI
    roi = frame[min_y : max_y + 1, min_x : max_x + 1].copy()

    roi = roi * mask_expanded
    # print(f"ROI shape: {roi.shape}, Mask Expanded shape: {mask_expanded.shape}")

    # print(roi.shape[0:2] == mask_expanded.shape[0:2])

    return roi


def mean_intensities_rgb(roi: np.array) -> np.array:
    return np.mean(roi, axis=(0, 1))


def calculate_head_pose(
    landmarks_extraceted: list, world_coords: list, frame_shape
) -> list:
    cam_matrix = np.array(
        [
            [1.0 * frame_shape[1], frame_shape[0] / 2, 0.0],
            [0.0, 1.0 * frame_shape[1], frame_shape[1] / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype="double",
    )
    camera_distortion = np.zeros((4, 1))
    world_coords = np.array(world_coords, dtype=np.float64)
    landmarks_extraceted = np.array(landmarks_extraceted, dtype=np.float64)
    _, rotation_vec, translation_vec = cv2.solvePnP(
        world_coords, landmarks_extraceted, cam_matrix, camera_distortion
    )
    rmat, jac = cv2.Rodrigues(rotation_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    print(type(angles))
    angles = np.array(angles, dtype=np.float64)
    angles = (angles - 180) * 360 / (2 * np.pi)
    print(angles)
    return angles


def headpose_landmarks(
    landmarks: list, landmarks_dict: dict, image_shape: list
) -> list:
    lm = landmarks
    lm_pose = landmarks_dict["head_pose_points"]
    head_landmarks_coords = np.array(
        [
            [int(lm[i].x * image_shape[1]), int(lm[i].y * image_shape[0])]
            for i in lm_pose
        ]
    )
    return head_landmarks_coords


def face_processing(control_pass) -> (np.array, np.array, list):
    control_obj = control_pass

    mp_face_mesh = mp.solutions.face_mesh
    prev_time = 0
    settings = toml.load("settings.toml")
    landmark_dict = settings["facemask"]
    world_coords = landmark_dict["map_face_base"]
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.8, min_tracking_confidence=0.9
    ) as face_mesh:
        for frame in capture_frame():
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # process frame with face_mesh
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                # print("face detected \n")
                for face in results.multi_face_landmarks:
                    # print("face", "\n")
                    landmark_list = list(face.landmark)
                    mask, cutout = create_mask(
                        landmark_list, frame.shape, landmark_dict
                    )
                    roi = extract_polygonal_roi(frame, mask, cutout)
                    head_lm_coords = headpose_landmarks(
                        landmark_list, landmark_dict, frame.shape
                    )
                    pose_angles = calculate_head_pose(
                        head_lm_coords, world_coords, frame.shape
                    )
                    # print("pose_angles", pose_angles, "\n")
                    rgb_samples = mean_intensities_rgb(roi)
                    control_obj.update_samples(frame, roi, rgb_samples, pose_angles)
                    cv2.imshow("Roi", roi)

                    # Display FPS
                    cv2.putText(
                        frame,
                        f"FPS: {int(fps)}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    # cv2.imshow("MediaPipe Face Mesh", frame)
                    #
                    # if cv2.waitKey(1) & 0xFF == ord("q"):
                    #    break
                    #
                    yield roi, frame, landmark_list


if __name__ == "__main__":
    signal_processor = Signal_processor()
    control_obj = control.Control(256)
    for frame in face_processing(control_obj):
        print("...")
