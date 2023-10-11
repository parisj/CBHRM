import cv2
from threading import Thread


class VideoStream:
    def __init__(self, src=0):
        WIDTH = 1280
        HEIGHT = 1080

        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while self.stream.isOpened() and not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                print("No frame")
                break
            yield frame

    def stop(self):
        self.stopped = True
        self.stream.release()
        cv2.destroyAllWindows()


# Usage
if __name__ == "__main__":
    video_stream = VideoStream().start()

    for frame in video_stream.update():
        cv2.imshow("Frame", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            video_stream.stop()
            break
