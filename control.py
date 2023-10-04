from collections import deque
from typing import List
import numpy as np
from blackboard import Blackboard
from dashboard import run_dash_app
from signal_processor import Signal_processor
import image_processor as ip
import threading


class Control:
    def __init__(
        self,
        blackboard: Blackboard,
        max_len: int,
        signal_processor: Signal_processor,
    ) -> None:
        self.blackboard = blackboard
        self.max_len = max_len
        self.signal_processor = signal_processor
        signal_processor.attach(self)
        self.signal_processor.attach(self)

    def attach(self, signal_processor: Signal_processor) -> None:
        self.signal_processor = signal_processor

    def update_samples(
        self, frame: np.array, roi: np.array, sample_rgb: np.array, head_pose: list
    ) -> None:
        self.blackboard.update_frame(frame)
        self.blackboard.update_roi(roi)
        self.blackboard.update_samples_rgb(sample_rgb)
        self.blackboard.update_samples_head_pose(head_pose)

    def get_samples(self) -> tuple:
        samples = self.blackboard.get_samples()
        return samples


if __name__ == "__main__":
    blackboard = Blackboard(256)
    signal_processor = Signal_processor()
    control_obj = Control(blackboard, 256, signal_processor)
    dash_thread = threading.Thread(target=run_dash_app, args=(control_obj,))
    dash_thread.start()
    for i in ip.face_processing(control_obj):
        pass
