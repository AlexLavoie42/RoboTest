"""perception_yolo_node.py

Real‑time YOLOv8 object‑detection publisher for the RoboTest project.

This ROS 2 node ingests RGB frames from an Isaac Sim camera, performs
object detection with Ultralytics YOLOv8, and publishes the results to
Redis Streams under the topic `sensory_data.vision`.

Requirements
------------
- ROS 2 Foxy (rclpy)
- ultralytics>=8.1.0 (pip install ultralytics)
- opencv‑python
- redis==5.*
- cv‑bridge==1.* (for ROS Image → NumPy conversion)

Usage
-----
```bash
ros2 run perception yolo_node --ros-args -p model_path:=yolov8n.pt -p redis_host:=127.0.0.1
```
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import redis
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image  # type: ignore
from ultralytics import YOLO  # type: ignore

# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class Detection:
    """Single object detection record."""

    id: str
    cls: str
    bbox: List[float]  # [x1, y1, x2, y2] normalized 0‑1
    confidence: float

    def to_json(self) -> str:  # noqa: D401
        return json.dumps(asdict(self))


# -----------------------------------------------------------------------------
# YOLO Node
# -----------------------------------------------------------------------------


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")

        # Parameters ---------------------------------------------------------
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("redis_host", "127.0.0.1")
        self.declare_parameter("redis_port", 6379)
        self.declare_parameter("camera_topic", "/rgb")
        self.declare_parameter("stream_key", "sensory_data.vision")

        model_path = str(self.get_parameter("model_path").value)
        redis_host = str(self.get_parameter("redis_host").value)
        redis_port = int(self.get_parameter("redis_port").value)
        camera_topic = str(self.get_parameter("camera_topic").value)
        self.stream_key: str = str(self.get_parameter("stream_key").value)

        # Load YOLO model -----------------------------------------------------
        if not Path(model_path).exists():
            self.get_logger().fatal(f"Model file not found: {model_path}")
            raise FileNotFoundError(model_path)
        self.model = YOLO(model_path)
        self.get_logger().info(f"Loaded YOLO model from {model_path}")

        # Redis client --------------------------------------------------------
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        try:
            self.redis.ping()
            self.get_logger().info(f"Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError as exc:
            self.get_logger().fatal(f"Failed to connect to Redis: {exc}")
            raise

        # ROS subscriptions ---------------------------------------------------
        self.bridge = CvBridge()
        self.create_subscription(Image, camera_topic, self.image_cb, 10)
        self.get_logger().info(f"Subscribed to camera topic: {camera_topic}")

    # ---------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------

    def image_cb(self, msg: Image) -> None:  # noqa: D401, N802
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detections = self.run_inference(frame)

        payload = {
            "timestamp": int(time.time() * 1000),
            "detections": [json.loads(det.to_json()) for det in detections],
        }
        # Add to Redis stream (maxlen capped for memory)
        self.redis.xadd(self.stream_key, {"data": json.dumps(payload)}, maxlen=5000, approximate=True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference(self, frame: np.ndarray) -> List[Detection]:
        height, width = frame.shape[:2]
        results = self.model.predict(frame, verbose=False, device=0)  # type: ignore[arg‑type]
        detections: List[Detection] = []
        for idx, row in enumerate(results[0].boxes.data.tolist()):
            x1, y1, x2, y2, conf, cls = row  # type: ignore[assignment]
            detections.append(
                Detection(
                    id=f"det-{int(time.time()*1e6)}-{idx}",
                    cls=self.model.names[int(cls)],
                    bbox=[x1 / width, y1 / height, x2 / width, y2 / height],
                    confidence=float(conf),
                )
            )
        # Optional: publish debug image with boxes
        self.publish_debug_image(frame, detections)
        return detections

    # ------------------------------------------------------------------
    def publish_debug_image(self, frame: np.ndarray, dets: List[Detection]):
        for det in dets:
            x1, y1, x2, y2 = [int(c) for c in (
                det.bbox[0] * frame.shape[1],
                det.bbox[1] * frame.shape[0],
                det.bbox[2] * frame.shape[1],
                det.bbox[3] * frame.shape[0],
            )]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det.cls}:{det.confidence:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Publishing via image_transport / debug topic could go here


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def main(args=None):  # noqa: D401
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
