import threading
import time
from openant.easy.node import Node
from openant.devices import ANTPLUS_NETWORK_KEY
from openant.devices.heart_rate import HeartRate, HeartRateData
import numpy as np
import toml
import csv


class HeartRateState:
    def __init__(self):
        self.beat_time = None
        self.beat_count = None
        self.heart_rate = None
        self.operating_time = None
        self.manufacturer_id_lsb = None
        self.serial_number = None
        self.previous_heart_beat_time = None
        self.battery_percentage = None
        self.BOOL_REFERENCE = False

    def update(self, data: HeartRateData):
        self.beat_time = data.beat_time
        self.beat_count = data.beat_count
        self.heart_rate = data.heart_rate
        self.operating_time = data.operating_time
        self.manufacturer_id_lsb = data.manufacturer_id_lsb
        self.serial_number = data.serial_number
        self.previous_heart_beat_time = data.previous_heart_beat_time
        self.battery_percentage = data.battery_percentage
        self.BOOL_REFERENCE = self.control.get_bool_reference()


def rec_HR(hr_state, lock, node, device):
    def on_found():
        print(f"Device {device} found and receiving")

    def on_device_data(page, page_name, data):
        if isinstance(data, HeartRateData):
            with lock:
                hr_state.beat_time = data.beat_time
                hr_state.beat_count = data.beat_count
                hr_state.heart_rate = data.heart_rate
                hr_state.operating_time = data.operating_time
                hr_state.manufacturer_id_lsb = data.manufacturer_id_lsb
                hr_state.serial_number = data.serial_number
                hr_state.previous_heart_beat_time = data.previous_heart_beat_time
                hr_state.battery_percentage = data.battery_percentage

    device.on_found = on_found
    device.on_device_data = on_device_data

    try:
        node.start()
    except Exception as exception:
        print(f"An error occurred in ANT+ communication: {exception}")
    finally:
        try:
            if node.channels:
                device.close_channel()
            node.stop()
        except NotImplementedError:
            print("attach_kernel_driver not implemented; continuing.")


class HeartRateMonitor:
    def __init__(self, e, stop_event):
        self.control = None
        self.hr_state = HeartRateState()
        self.lock = threading.Lock()
        self.collection_thread = None
        self.running = False
        self.captured_data = []
        self.event = e
        self.stop_event = stop_event
        self.settings = toml.load("settings.toml")
        self.settings_reference = self.settings["reference_measurement"]
        self.time_window = self.settings_reference["hrv_window"]
        self.fps = self.settings["filtering"]["fs"]
        self.device_id = self.settings["reference_measurement"]["device_id"]
        self.hrv_ref = None
        self.hr_ref = None
        self.last_rmssd = None
        self.last_timestamps = None
        self.last_beat_time = None

    def set_control(self, control_obj):
        self.control = control_obj

    def calculate_hrv(self):
        """
        Calculate the heart rate variability (HRV) using RMSSD method within a time window.

        :param data: List of dictionaries containing 'beat_time' and 'time' (timestamp).
        :param time_window: Time window in seconds for calculating HRV.
        :return: Calculated HRV value.
        """
        data, offsetmd = self.control.blackboard.get_monitoring_data()

        # Extract beat times and timestamps
        try:
            beat_times = np.array([d["beat_time"] for d in data])

        except TypeError:
            return 0

        timestamps = np.array([d["time"] for d in data])
        # Remove duplicates in beat_times while keeping the corresponding timestamps
        _, unique_indices = np.unique(beat_times, return_index=True)

        sorted_unique_indices = np.sort(unique_indices)
        # sorted_unique_indices =unique_indices
        unique_beat_times = beat_times[sorted_unique_indices]
        unique_timestamps = timestamps[sorted_unique_indices]

        # Find indexes of beats within the time window
        current_time = unique_timestamps[-1]
        valid_indices = unique_timestamps >= (
            max(0, current_time - self.time_window / self.fps)
        )

        # Calculate differences of valid beat times using np.diff
        diffs = np.diff(unique_beat_times[valid_indices])
        # handle overflow
        valid_diffs = np.where(diffs < 0, diffs + 64.0, diffs) * 1000

        # Calculate RMSSD
        if len(valid_diffs) < 2:
            return 0  # Not enough data to calculate HRV
        squared_diffs = np.square(valid_diffs)
        rmssd = np.sqrt(np.mean(squared_diffs))
        if self.last_rmssd is not None and self.last_rmssd + 30 < rmssd:
            print()
        self.last_rmssd = rmssd
        self.last_beat_time = diffs

        return rmssd

    def start_collection(self):
        """Starts the heart rate data collection if not already running."""
        if not self.running:
            self.running = True
            self.collection_thread = threading.Thread(target=self._collect_data)
            self.collection_thread.start()

    def stop_collection(self):
        """Stops the heart rate data collection."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
            self.collection_thread = None
        self.reset_attributes()

    def _collect_data(self):
        """Internal method to handle the data collection."""
        node = Node()
        node.set_network_key(0x00, ANTPLUS_NETWORK_KEY)
        device = HeartRate(node, device_id=self.device_id)
        rec_thread = threading.Thread(
            target=rec_HR, args=(self.hr_state, self.lock, node, device)
        )
        rec_thread.start()
        value = False
        try:
            while not self.stop_event.is_set() and self.running:
                self.event.wait()
                with self.lock:
                    if (
                        self.hr_state.beat_time is not None
                        and self.hr_state.heart_rate is not None
                    ):
                        current_state = {
                            "beat_time": self.hr_state.beat_time,
                            "beat_count": self.hr_state.beat_count,
                            "heart_rate": self.hr_state.heart_rate,
                            "operating_time": self.hr_state.operating_time,
                            "manufacturer_id_lsb": self.hr_state.manufacturer_id_lsb,
                            "serial_number": self.hr_state.serial_number,
                            "previous_heart_beat_time": self.hr_state.previous_heart_beat_time,
                            "battery_percentage": self.hr_state.battery_percentage,
                            "time": time.time(),
                        }
                        if value:
                            self.hrv_ref = self.calculate_hrv()
                            current_state["heart_rate_variability"] = self.hrv_ref
                            self.control.blackboard.update_hrv_reference(self.hrv_ref)
                        self.captured_data.append(current_state)
                        self.control.update_monitoring_data(self.captured_data)
                        self.control.blackboard.update_hr_reference(
                            self.hr_state.heart_rate
                        )
                        value = True
                self.event.clear()

        finally:
            if node.channels:
                device.close_channel()
            node.stop()

    def reset_attributes(self):
        """Resets the attributes of the blackboard. (restarting the application)"""
        self.control.blackboard.reset_attributes()


class HeartRateEvaluation:
    def __init__(self, control_obj) -> None:
        self.settings = toml.load("settings.toml")
        self.current_state = None
        self.captured_data = []
        self.control = control_obj
        self.heartratemonitor = HeartRateMonitor(None, None)
        self.heartratemonitor.set_control(control_obj)
        self.hrv_ref = None
        self.value = False

    def extract_data_dataset(self, row):
        current_state = {
            "beat_time": float(row["beat_time"]),
            "beat_count": float(row["beat_count"]),
            "heart_rate": int(row["heart_rate"]),
            "operating_time": int(row["operating_time"]),
            "manufacturer_id_lsb": row["manufacturer_id_lsb"],
            "serial_number": row["serial_number"],
            "previous_heart_beat_time": float(row["previous_heart_beat_time"]),
            "battery_percentage": int(row["battery_percentage"]),
            "time": time.time(),
        }
        if self.value:
            self.hrv_ref = self.heartratemonitor.calculate_hrv()
            current_state["heart_rate_variability"] = self.hrv_ref
            self.control.blackboard.update_hrv_reference(self.hrv_ref)
        self.captured_data.append(current_state)
        self.control.update_monitoring_data(self.captured_data)
        self.control.blackboard.update_hr_reference(int(row["heart_rate"]))
        self.value = True

    def read_csv_to_list(self, file_path):
        """
        Read a CSV file into a list of dictionaries, including empty rows.

        Parameters:
        file_path (str): The path to the CSV file.

        Returns:
        list: A list of dictionaries, each representing a row in the CSV file.
        """
        with open(file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return [row for row in reader]
