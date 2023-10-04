import numpy as np
from collections import deque
import cv2


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

    def calc_samples_rgb(self, image: np.array) -> None:
        self.samples_rgb = average_intensities_rgb(image)

    def calc_pos(self, samples_head_pose: deque, samples_rgb: deque) -> float:
        # TODO Implement POS calculation
        # self.sample_pos = pos
        pass

    def update(self, image: np.array, landmark) -> None:
        self.calc_samples_rgb(image)
        self.calc_samples_head_pose(image, landmark)
        self.control.update()
