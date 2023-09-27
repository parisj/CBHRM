import cv2
import toml
import mediapipe as mp
import time
import numpy as np


def capture_frame() -> np.array:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        yield frame
    cap.release()
    cv2.destroyAllWindows()
    return 0


def create_mask(lm: np.array, image_shape: np.array, mask_lm: dict) -> (np.array, list):
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

    return mask, cutout


def extract_polygonal_roi(frame: np.array, mask: np.array, cutout: list) -> np.array:
    min_x, max_x, min_y, max_y = cutout
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask_expanded = np.expand_dims(mask, axis=2)

    roi = frame[min_y:max_y, min_x:max_x] * mask_expanded[min_y:max_y, min_x:max_x]

    return roi


def face_processing() -> (np.array, np.array, list):
    mp_face_mesh = mp.solutions.face_mesh
    prev_time = 0
    settings = toml.load("settings.toml")
    landmark_dict = settings["facemask"]

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
                for face in results.multi_face_landmarks:
                    landmark_list = list(face.landmark)
                    mask, cutout = create_mask(landmark_dict, frame.shape)
                    roi = extract_polygonal_roi(frame, mask, cutout)
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
            cv2.imshow("MediaPipe Face Mesh", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    yield roi, frame, landmarks_list


if __name__ == "__main__":
    face_processing()
