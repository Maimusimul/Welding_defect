import numpy as np
import cv2
import pytest
from core.image_processor import ImageProcessor


def test_detect_defects_runs_without_errors():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    processor = ImageProcessor("models/YOLO_11.pt")
    annotated, detections = processor.detect_defects(img)

    assert isinstance(annotated, np.ndarray)
    assert isinstance(detections, list)
