import numpy as np
from collections import deque
import cv2
from observer import Observer
from collections import deque

"""
TODO: 
- [ ] implement signal processing
   | - [ ] Resampling, Interpolation
- [ ] implement POS algorithm
- [ ] implement 3D Head Pose Estimation
   | - [ ] Pitch 
   | - [ ] Roll
   | - [ ] yaw
- [ ] Integrate Observer pattern 
- [ ] State papern? 
"""


class Signal_processor:
    def __init__(self, length: int) -> None:
        self.observer = Observer(length)
        observer.attach(self)

        self.length = length
        self.samples_rgb = None
        self.sample_head_pose = None
        self.sample_pos = None
        self.pose_head = None

    def average_intensities_rgb(self, image: np.array) -> np.array:
        self.sample_rgb.average(image, axis=(0, 1))

    def calc_samples_rgb(self, image: np.array) -> None:
        self.samples_rgb = average_intensities_rgb(image)

    def calc_pitch(self, landmarks: np.array) -> float:
        # TODO Implement pitch calculation
        # self.sample_head_pose = pitch
        pass

    def calc_roll(self, landmarks: np.array) -> float:
        # TODO Implement roll calculation
        # self.sample_head_pose = roll
        pass

    def calc_yaw(self, landmarks: np.array) -> float:
        # TODO Implement yaw calculation
        # self.sample_head_pose = yaw
        pass

    def calc_samples_head_pose(self) -> float:
        # TODO Implement 3D head pose calculation
        # https://medium.com/@susanne.thierfelder/head-pose-estimation-with-mediapipe-and-opencv-in-javascript-c87980df3acb
        # Landmarks [33, 263, 1, 61, 291, 199]
        # self.sample_head_pose = head_pose
        pass

    def calc_pos(self, samples_head_pose: deque, samples_rgb: deque) -> float:
        # TODO Implement POS calculation
        # self.sample_pos = pos
        pass

    def update(self, image: np.array, landmark) -> None:
        self.calc_samples_rgb(image)
        self.calc_samples_head_pose(image, landmark)
        self.observer.update()
