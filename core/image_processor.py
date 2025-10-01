import cv2
import numpy as np
import os
from ultralytics import YOLO


class ImageProcessor:
    def __init__(self, model_path: str = "models/YOLO_11.pt"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
            self.model = YOLO(self.model_path)

    def detect_defects(self, image):
        self.load_model()  # ✅ добавляем вызов
        results = self.model(image)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                    label = self.model.names.get(cls_id, str(cls_id))  # <-- строковое имя класса

                    confidence = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    xyxy = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else [0, 0, 0, 0]

                    detections.append({
                        "label": str(label),
                        "confidence": confidence,
                        "bbox": xyxy
                    })

        annotated = self._draw_boxes(image.copy(), detections)
        return annotated, detections

    def _draw_boxes(self, image, detections):
        def get_color(label, conf):
            label = label.lower()
            if "good" in label:
                return (0, 255, 0)  # зелёный
            elif conf > 0.7:
                return (0, 0, 255)  # красный
            else:
                return (0, 255, 255)  # жёлтый

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = det['label']
            conf = det['confidence']
            color = get_color(label, conf)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({conf:.2f})"
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image

    def get_color(label, conf):
        label = label.lower()
        if "good" in label:
            return (0, 255, 0)  # зелёный
        elif conf > 0.7:
            return (0, 0, 255)  # красный — высокая уверенность дефекта
        else:
            return (0, 255, 255)  # жёлтый — менее уверенный дефект


