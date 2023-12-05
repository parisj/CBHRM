import threading
import time
from openant.easy.node import Node
from openant.devices import ANTPLUS_NETWORK_KEY
from openant.devices.heart_rate import HeartRate, HeartRateData


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

    def set_control(self, control_obj):
        self.control = control_obj

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
        device = HeartRate(node, device_id=0)
        rec_thread = threading.Thread(
            target=rec_HR, args=(self.hr_state, self.lock, node, device)
        )
        rec_thread.start()

        try:
            while not self.stop_event.is_set() and self.running:
                self.event.wait()
                with self.lock:
                    current_state = {
                        "beat_time": self.hr_state.beat_time,
                        "beat_count": self.hr_state.beat_count,
                        "heart_rate": self.hr_state.heart_rate,
                        "operating_time": self.hr_state.operating_time,
                        "manufacturer_id_lsb": self.hr_state.manufacturer_id_lsb,
                        "serial_number": self.hr_state.serial_number,
                        "previous_heart_beat_time": self.hr_state.previous_heart_beat_time,
                        "battery_percentage": self.hr_state.battery_percentage,
                    }
                    self.captured_data.append(current_state)
                    self.control.update_monitoring_data(self.captured_data)
                self.event.clear()
        finally:
            device.close_channel()
            node.stop()

    def reset_attributes(self):
        """Resets the attributes of the blackboard. (restarting the application)"""
        self.control.blackboard.reset_attributes()
