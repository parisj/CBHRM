import cv2
import threading
import csv
from openant.easy.node import Node
from openant.devices import ANTPLUS_NETWORK_KEY
from openant.devices.heart_rate import HeartRate, HeartRateData
import cProfile
import toml
import os
import time


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

    def update(self, data: HeartRateData):
        self.beat_time = data.beat_time
        self.beat_count = data.beat_count
        self.heart_rate = data.heart_rate
        self.operating_time = data.operating_time
        self.manufacturer_id_lsb = data.manufacturer_id_lsb
        self.serial_number = data.serial_number
        self.previous_heart_beat_time = data.previous_heart_beat_time
        self.battery_percentage = data.battery_percentage


def rec_HR(device_id, hr_state, lock, node, device):
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
    except Exception as e:
        print(f"An error occurred in ANT+ communication: {e}")
    finally:
        try:
            device.close_channel()
            node.stop()
        except NotImplementedError:
            print("attach_kernel_driver not implemented; continuing.")


def main():
    # Define the GStreamer pipeline
    settings = toml.load("settings.toml")
    fps = settings["record_data_set"]["fps_camera"]
    width = settings["record_data_set"]["resolution"][0]
    height = settings["record_data_set"]["resolution"][1]
    # device_id = settings["record_data_set"]["device_id"]

    video_path = settings["record_data_set"]["video_path"]
    readings_path = settings["record_data_set"]["readings_path"]
    video_path = os.path.join(os.getcwd(), video_path)
    readings_path = os.path.join(os.getcwd(), readings_path)
    gstreamer_pipeline = (
        'mfvideosrc device-name="HD Pro Webcam C920" ! '
        "video/x-raw, format=NV12, width="
        + str(width)
        + ", height="
        + str(height)
        + ", pixel-aspect-ratio=1/1, framerate="
        + str(fps)
        + "/1,  colorimetry=1:4:0:1 ! "
        "videoconvert ! "  # Convert to a format compatible with appsink, if necessary
        "appsink"
    )

    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    out = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps, (width, height)
    )

    hr_state = HeartRateState()
    lock = threading.Lock()
    node = Node()
    node.set_network_key(0x00, ANTPLUS_NETWORK_KEY)
    device = HeartRate(node, device_id=0)

    hr_thread = threading.Thread(target=rec_HR, args=(0, hr_state, lock, node, device))
    hr_thread.start()

    captured_data = []
    start = time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)

            with lock:
                current_state = {
                    "beat_time": hr_state.beat_time,
                    "beat_count": hr_state.beat_count,
                    "heart_rate": hr_state.heart_rate,
                    "operating_time": hr_state.operating_time,
                    "manufacturer_id_lsb": hr_state.manufacturer_id_lsb,
                    "serial_number": hr_state.serial_number,
                    "previous_heart_beat_time": hr_state.previous_heart_beat_time,
                    "battery_percentage": hr_state.battery_percentage,
                }
                print(current_state)
                captured_data.append(current_state)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                # Properly release resources and save data
                cap.release()
                out.release()
                cv2.destroyAllWindows()

                keys = captured_data[0].keys()
                with open(readings_path, "w", newline="") as output_file:
                    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(captured_data)
                print("Duration Recording:", start - time.time())
                device.close_channel()
                node.stop()
                hr_thread.join()
                print("Exiting...")
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Properly release resources and save data
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        hr_thread.join()

        keys = captured_data[0].keys()
        with open(readings_path, "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(captured_data)


if __name__ == "__main__":
    main()
