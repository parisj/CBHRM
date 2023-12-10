import cv2
from threading import Thread


class VideoStream:
    def __init__(self, e, stop_event):
        # gstreamer_pipeline = (
        #    'mfvideosrc device-name="DICOTA 4K" ! '
        #    "video/x-raw, format=NV12, width=1280, height=720, pixel-aspect-ratio=1/1, framerate=30/1 ! "
        #    "videoconvert ! "  # Convert to a format compatible with appsink, if necessary
        #    "videorate ! video/x-raw, framerate=30/1 ! "
        #    "appsink max-buffers=2 drop=true"
        # )

        gstreamer_pipeline = (
            'mfvideosrc device-name="HD Pro Webcam C920" ! '
            "video/x-raw, format=NV12, width=1600, height=896, pixel-aspect-ratio=1/1, framerate=20/1 ! "
            "videoconvert ! "  # Convert to a format compatible with appsink, if necessary
            "appsink"
        )

        self.stream = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        WIDTH, HEIGHT = 1600, 896

        self.event = e
        self.stop_event = stop_event

    #    WIDTH = 1280
    #    HEIGHT = 1080

    #    self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    #    self.stream.set(cv2.CAP_PROP_FPS, 30)
    #    self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    #    self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    #    self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    #    self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    #    self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while self.stream.isOpened() and not self.stop_event.is_set():
            ret, frame = self.stream.read()
            self.event.set()
            if not ret:
                print("No frame")
                break
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            yield frame
        self.stop()

    def stop(self):
        self.stream.release()
        cv2.destroyAllWindows()


class VideoPlayer:
    def __init__(self, video_file):
        self.video_file = video_file
        self.stopped = False

    def start(self):
        self.stopped = False
        self.play_thread = threading.Thread(target=self.play_video, args=())
        self.play_thread.start()
        return self

    def play_video(self):
        cap = cv2.VideoCapture(self.video_file)

        while not self.stopped:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break

        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.stopped = True
        self.play_thread.join()
        cv2.destroyAllWindows()


# Usage
if __name__ == "__main__":
    video_stream = VideoStream().start()

    for frame in video_stream.update():
        cv2.imshow("Frame", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            video_stream.stop()
            break
