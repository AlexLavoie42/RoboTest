"""tests/test_yolo_node.py

Unit tests for perception_yolo_node.py
Requires pytest, pytest‑asyncio, and monkeypatch fixtures.
The tests stub out heavy dependencies (ROS, Redis, GPU) so they run in CI.
"""

import json
import types
from unittest import mock

import numpy as np
import pytest

# We will monkey‑patch YOLO and rclpy Node to avoid GPU use


@pytest.fixture(autouse=True)
def patch_heavy(monkeypatch):
    # Stub YOLO.predict → returns fake detections
    class DummyResults:
        boxes = types.SimpleNamespace(data=[[0, 0, 10, 10, 0.9, 1]])

    class DummyModel:
        names = {1: "dummy"}

        def __init__(self, *_, **__):
            pass

        def predict(self, frame, verbose=False, device=0):  # noqa: D401
            return [DummyResults()]

    monkeypatch.setitem(__import__("sys").modules, "ultralytics", mock.MagicMock(YOLO=DummyModel))

    # Patch rclpy Node base class to remove ROS dependency
    dummy_node_base = type("DummyNodeBase", (), {"__init__": lambda self, *_, **__: None, "get_logger": lambda self: mock.MagicMock(info=lambda *_: None, fatal=lambda *_: None)})
    monkeypatch.setitem(__import__("sys").modules, "rclpy", mock.MagicMock(init=lambda *_, **__: None, spin=lambda *_: None, shutdown=lambda: None, Node=dummy_node_base))

    # Patch redis
    monkeypatch.setitem(__import__("sys").modules, "redis", mock.MagicMock(Redis=lambda **_: mock.MagicMock(xadd=lambda *_, **__: None, ping=lambda: True)))


def test_detection_json_serialization():
    from perception.perception_yolo_node import Detection

    det = Detection(id="1", cls="cup", bbox=[0, 0, 1, 1], confidence=0.99)
    data = json.loads(det.to_json())
    assert data["cls"] == "cup"
    assert 0 <= data["confidence"] <= 1


def test_run_inference(monkeypatch):
    from perception.perception_yolo_node import YoloNode

    # Patch image processing functions
    monkeypatch.setattr(YoloNode, "publish_debug_image", lambda *_: None)

    node = YoloNode()

    dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    detections = node.run_inference(dummy_frame)

    assert len(detections) == 1
    assert detections[0].cls == "dummy"
    assert 0.0 <= detections[0].confidence <= 1.0
