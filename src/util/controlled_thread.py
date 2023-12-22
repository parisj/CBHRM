import threading
import time


class ControlledThread(threading.Thread):
    def __init__(self, thread_registry):
        super().__init__()
        self._stop_flag = threading.Event()
        self.thread_registry = thread_registry
        self.thread_registry.append(self)

    def stop(self):
        self._stop_flag.set()

    def run(self):
        raise NotImplementedError("Subclasses should implement this!")


class WorkerThread(ControlledThread):
    def run(self):
        while not self._stop_flag.is_set():
            print(f"Working in {self.name}")
            time.sleep(1)
            # Example of spawning a child thread
            if self.name == "Worker-1" and not any(
                t.name == "Worker-1-Child" for t in self.thread_registry
            ):
                ChildThread(self.thread_registry).start()
        print(f"{self.name} stopping")


class ChildThread(ControlledThread):
    def run(self):
        self.name = "Worker-1-Child"
        while not self._stop_flag.is_set():
            print(f"Working in {self.name}")
            time.sleep(1)
        print(f"{self.name} stopping")


# Thread registry to keep track of all threads
thread_registry = []

# Start threads
for _ in range(2):
    WorkerThread(thread_registry).start()

# Allow threads to run for a bit
time.sleep(5)

# Signal all threads to stop
for thread in thread_registry:
    thread.stop()

# Wait for all threads to finish
for thread in thread_registry:
    thread.join()

print("All threads have been stopped and joined")
