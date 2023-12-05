from collections import deque
from typing import List
import numpy as np
from blackboard import Blackboard
from dashboard import run_dash_app
from signal_processor import Signal_processor
import image_processor as ip
import threading
from reference_meas import HeartRateMonitor
import csv
from viztracer import VizTracer


class Control:
    def __init__(
        self,
        blackboard: Blackboard,
        max_len: int,
        signal_processor: Signal_processor,
        hr_monitor: HeartRateMonitor,
    ) -> None:
        self.blackboard = blackboard
        self.max_len = max_len
        self.signal_processor = signal_processor
        signal_processor.attach(self)
        self.signal_processor.attach(self)
        self.hr_monitor = hr_monitor

    def attach(self, signal_processor: Signal_processor) -> None:
        self.signal_processor = signal_processor

    def update_samples(
        self,
        frame: np.array,
        roi: np.array,
        sample_rgb: np.array,
        head_pose: list,
        time_stamp: float,
    ) -> None:
        self.blackboard.update_frame(frame)
        self.blackboard.update_roi(roi)
        self.blackboard.update_samples_rgb(sample_rgb)
        self.blackboard.update_samples_head_pose(head_pose)
        self.blackboard.update_time_stamp(time_stamp)

    def update_monitoring_data(self, current_state: dict) -> None:
        self.blackboard.update_monitoring_data(current_state)

    def get_monitoring_data(self) -> dict:
        return self.blackboard.get_monitoring_data()

    def get_samples(self) -> tuple:
        samples = self.blackboard.get_samples()
        return samples

    def get_signal_samples(self) -> tuple:
        samples = self.blackboard.get_signal_samples()
        return samples

    def get_samples_rgb(self) -> tuple:
        samples = self.blackboard.get_samples_rgb()
        return samples

    def get_samples_head_pose(self) -> tuple:
        samples = self.blackboard.get_samples_head_pose()
        return samples

    def get_samples_rPPG(self) -> tuple:
        samples = self.blackboard.get_samples_rPPG()
        return samples

    def get_samples_rhythmic(self) -> tuple:
        samples = self.blackboard.get_samples_rhythmic()
        return samples

    def get_bool_reference(self) -> bool:
        return self.blackboard.get_bool_reference()

    def write_results(self, path, write_event, stop_event) -> None:
        # THREAD HANDLING!! Combine with thread controller
        while not stop_event.is_set():
            write_event.wait()
            peaks = self.blackboard.get_peaks()
            diff_peaks = self.blackboard.get_diff_peaks()
            hr = self.blackboard.get_hr()
            hrv = self.blackboard.get_rmssd()
            with open(path, "a", newline="") as file:
                writer = csv.writer(file)
                # Check if the file is empty to write the header
                if file.tell() == 0:
                    writer.writerow(
                        [
                            "Peaks",
                            "Difference Between Peaks",
                            "Heart Rate",
                            "Heart Rate Variability",
                        ]
                    )

                # Write data
                writer.writerow(
                    [
                        ";".join(map(str, peaks)),
                        ";".join(map(str, diff_peaks)),
                        hr[-1],
                        hrv,
                    ]
                )
            write_event.clear()


if __name__ == "__main__":
    # tracer = VizTracer()

    stop_event = threading.Event()
    took_sample_event = threading.Event()
    write_event = threading.Event()
    initial_samples_event = threading.Event()
    new_sample_event = threading.Event()

    blackboard = Blackboard(256, initial_samples_event)
    signal_processor = Signal_processor(write_event)
    hr_monitor = HeartRateMonitor(took_sample_event, stop_event)

    control_obj = Control(blackboard, 256, signal_processor, hr_monitor)
    hr_monitor.set_control(control_obj)

    dash_thread = threading.Thread(
        target=run_dash_app, args=(control_obj, hr_monitor, stop_event)
    )
    write_thread = threading.Thread(
        target=control_obj.write_results, args=("results.csv", write_event, stop_event)
    )
    image_thread = threading.Thread(
        target=ip.face_processing,
        args=(control_obj, took_sample_event, stop_event, new_sample_event),
    )
    signal_processor_thread = threading.Thread(
        target=signal_processor.signal_processing_function,
        args=(stop_event, initial_samples_event, new_sample_event),
    )
    # tracer.start()
    image_thread.start()
    signal_processor_thread.start()

    dash_thread.start()
    write_thread.start()

    try:
        # Wait for a condition or user input to stop the threads
        input("Press Enter to stop all threads...")

    except KeyboardInterrupt:
        # Handle keyboard interrupts (Ctrl+C)
        print("Stopping threads due to keyboard interrupt...")

    finally:
        # Signal all threads to stop
        stop_event.set()
        # Wait for all threads to finish
        image_thread.join()
        write_thread.join()
        signal_processor_thread.join()
        # tracer.stop()
        # tracer.save(output_file="result.json")
        dash_thread.join()
        # stop_dash_app()

        print("All threads have been stopped.")
