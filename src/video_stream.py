import cv2
from threading import Thread
import toml
import time
from viztracer import VizTracer


class VideoStream:
    def __init__(self, e, stop_event):
        # gstreamer_pipeline = (
        #    'mfvideosrc device-name="DICOTA 4K" ! '
        #    "video/x-raw, format=NV12, width=1280, height=720, pixel-aspect-ratio=1/1, framerate=30/1 ! "
        #    "videoconvert ! "  # Convert to a format compatible with appsink, if necessary
        #    "videorate ! video/x-raw, framerate=30/1 ! "
        #    "appsink max-buffers=2 drop=true"
        # )
        self.settings = toml.load("settings.toml")
        self.settings_camera = self.settings["camera"]
        self.fps = self.settings_camera["fps_camera"]
        self.width = self.settings_camera["resolution"][0]
        self.height = self.settings_camera["resolution"][1]
        self.video_source = self.settings["video_source"]
        self.settings_video_source = self.settings["video_source"]
        self.path_read = self.settings_video_source["path"]
        self.event = e
        self.stop_event = stop_event
        self.live = self.settings_video_source["live"]
        self.path_write = self.settings["record_data_set"]["video_path"]
        self.out = None

        if self.live:
            gstreamer_pipeline = (
                'mfvideosrc device-name="HD Pro Webcam C920" ! '
                "video/x-raw, format=NV12, width="
                + str(self.width)
                + ", height="
                + str(self.height)
                + ", pixel-aspect-ratio=1/1, framerate="
                + str(self.fps)
                + "/1,  colorimetry=1:4:0:1 ! "
                "videoconvert ! "  # Convert to a format compatible with appsink, if necessary
                "appsink"
            )

            self.stream = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
            self.out = cv2.VideoWriter(
                self.path_write,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                self.fps,
                (self.width, self.height),
            )
        else:
            self.stream = cv2.VideoCapture(self.path_read)
            self.stream.set(cv2.CAP_PROP_FPS, self.fps)
            self.stream.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
            )

    def start(self):
        if not self.settings_video_source["live"]:
            time.sleep(5)
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while self.stream.isOpened() and not self.stop_event.is_set():
            ret, frame = self.stream.read()
            self.event.set()
            if not ret:
                print("No frame")
                break
            if self.live:  # save video
                self.out.write(frame)
            cv2.imshow("Frame", frame)
            if self.settings_video_source["live"]:
                cv2.waitKey(1)
            else:
                cv2.waitKey(800 // self.fps)

            yield frame

        self.stop()

    def stop(self):
        if self.stream.isOpened():
            self.stream.release()
            if self.live:
                self.out.release()
                print("--- saved video under " + self.path_write + "---")
        cv2.destroyAllWindows()
