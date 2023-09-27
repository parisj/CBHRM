from collections import deque
from typing import List
import numpy as np
from signal_processor import Signal_processor


class Observer:
    def __init__(self, max_len: int) -> None:
        self.samples_rgb = deque(maxlen=max_len)
        self.samples_head_pose = deque(maxlen=max_len)
        self.samples_pos = deque(maxlen=max_len)
        self.signal_processor = None

    def attach(self, signal_processor: Signal_processor) -> None:
        self.signal_processor = signal_processor

    def update_samples_rgb(self, sample_rgb) -> None:
        self.samples_rgb.append(sample_rgb)

    def update_samples_head_pose(self, head_pose: float) -> None:
        self.samples_head_pose.append(head_pose)

    def update_samples_pos(self, pos: List[float]) -> None:
        self.samples_pos = signal_processor.calc_pos(
            self.samples_head_pose, self.samples_rgb
        )

    def get_samples(self) -> deque:
        return self.samples_rgb, self.samples_head_pose, self.samples_pos

    def update(self):
        self.update_samples_rgb(self.signal_processor.sample_rgb)
        self.update_samples_head_pose(self.signal_processor.sample_head_pose)
        self.update_samples_pos()
