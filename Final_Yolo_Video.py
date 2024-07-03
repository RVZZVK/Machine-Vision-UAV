import csv
import os
import queue
import tempfile
import threading
from ultralytics import YOLO
import olympe
from olympe.video.renderer import PdrawRenderer
import cv2
import numpy as np

olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")

Model_Path = r"/home/labpc/Downloads/yolov8n-seg.pt"
model = YOLO(Model_Path)
class StreamingExample:
    def __init__(self):
        self.drone = olympe.Drone(DRONE_IP)
        self.tempd = tempfile.mkdtemp(prefix="olympe_streaming_test_")
        print(f"Olympe streaming example output dir: {self.tempd}")
        self.h264_stats_file = open(os.path.join(self.tempd, "h264_stats.csv"), "w+")
        self.h264_stats_writer = csv.DictWriter(
            self.h264_stats_file, ["fps", "bitrate"]
        )
        self.h264_stats_writer.writeheader()
        self.frame_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.yuv_frame_processing)
        self.renderer = None

    def start(self):
        assert self.drone.connect(retry=3)

        if DRONE_RTSP_PORT is not None:
            self.drone.streaming.server_addr = f"{DRONE_IP}:{DRONE_RTSP_PORT}"

        self.drone.streaming.set_output_files(
            video=os.path.join(self.tempd, "streaming.mp4"),
            metadata=os.path.join(self.tempd, "streaming_metadata.json"),
        )

        self.drone.streaming.set_callbacks(
            raw_cb=self.yuv_frame_cb,
            h264_cb=self.h264_frame_cb,
            start_cb=self.start_cb,
            end_cb=self.end_cb,
            flush_raw_cb=self.flush_cb,
        )

        self.drone.streaming.start()
        self.running = True
        self.processing_thread.start()

    def stop(self):
        self.running = False
        self.processing_thread.join()
        if self.renderer is not None:
            self.renderer.stop()
        assert self.drone.streaming.stop()
        assert self.drone.disconnect()
        self.h264_stats_file.close()

    def yuv_frame_cb(self, yuv_frame):
        yuv_frame.ref()
        self.frame_queue.put_nowait(yuv_frame)

    def yuv_frame_processing(self):
        while self.running:
            try:
                yuv_frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # Convert YUV frame to OpenCV format
            info = yuv_frame.info()
            height, width = info["raw"]["frame"]["info"]["height"], info["raw"]["frame"]["info"]["width"]
            yuv_data = yuv_frame.as_ndarray()
            yuv_data = yuv_data.reshape((height * 3 // 2, width))
            bgr_frame = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)
            results = model(bgr_frame)
            # Plot boundary & image masking from obtained results in f2f format
            annotated_frame = results[0].plot()
            # Display the frame
            cv2.imshow('Machine Vision', annotated_frame)
            #cv2.imshow("Drone Feed", bgr_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            yuv_frame.unref()
        cv2.destroyAllWindows()

    def flush_cb(self, stream):
        if stream["vdef_format"] != olympe.VDEF_I420:
            return True
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait().unref()
        return True

    def start_cb(self):
        pass

    def end_cb(self):
        pass

    def h264_frame_cb(self, h264_frame):
        frame_pointer, frame_size = h264_frame.as_ctypes_pointer()

        info = h264_frame.info()
        frame_ts = info["ntp_raw_timestamp"]
        if not bool(info["is_sync"]):
            while len(self.h264_frame_stats) > 0:
                start_ts, _ = self.h264_frame_stats[0]
                if (start_ts + 1e6) < frame_ts:
                    self.h264_frame_stats.pop(0)
                else:
                    break
            self.h264_frame_stats.append((frame_ts, frame_size))
            h264_fps = len(self.h264_frame_stats)
            h264_bitrate = 8 * sum(map(lambda t: t[1], self.h264_frame_stats))
            self.h264_stats_writer.writerow({"fps": h264_fps, "bitrate": h264_bitrate})


def test_streaming():
    streaming_example = StreamingExample()
    streaming_example.start()

    # Adjust this time to test the live video feed (in seconds)
    test_duration = 500  # 70 seconds

    try:
        # Let the streaming run for the specified duration
        streaming_example.processing_thread.join(test_duration)
    except KeyboardInterrupt:
        pass

    streaming_example.stop()


if __name__ == "__main__":
    test_streaming()
