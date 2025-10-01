import os
import numpy as np
import cv2
import pytest
from core.image_processor import ImageProcessor

# ========== Тесты для ImageProcessor ==========

# 1. Проверка: ошибка, если файл модели не найден
def test_model_not_found_raises_error(tmp_path):
    fake_model_path = tmp_path / "no_model.pt"
    processor = ImageProcessor(str(fake_model_path))
    with pytest.raises(FileNotFoundError):
        processor.load_model()

# 2. Проверка: метод detect_defects возвращает аннотированное изображение и список детекций
def test_detect_defects_returns_result(monkeypatch):
    class FakeYOLO:
        names = {0: "good weld"}
        def __call__(self, image):
            class Box:
                cls = [np.array([0])]
                conf = [np.array([0.9])]
                xyxy = [np.array([10, 10, 100, 100])]
            class Result:
                boxes = [Box()]
            return [Result()]
    processor = ImageProcessor()
    processor.model = FakeYOLO()

    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    annotated, detections = processor.detect_defects(dummy_img)

    assert isinstance(annotated, np.ndarray)
    assert isinstance(detections, list)
    assert detections[0]['label'] == "good weld"

# 3. Проверка: зелёная рамка рисуется, если label содержит "good"
def test_draw_boxes_color_green():
    processor = ImageProcessor()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [{"label": "good weld", "confidence": 0.9, "bbox": [10, 10, 50, 50]}]
    out = processor._draw_boxes(dummy_img.copy(), detections)
    assert out.shape == dummy_img.shape

# 4. Проверка: красная рамка при высокой уверенности дефекта (>0.7)
def test_draw_boxes_color_red():
    processor = ImageProcessor()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [{"label": "defect", "confidence": 0.95, "bbox": [10, 10, 50, 50]}]
    out = processor._draw_boxes(dummy_img.copy(), detections)
    assert out.shape == dummy_img.shape

# 5. Проверка: жёлтая рамка при низкой уверенности дефекта (<=0.7)
def test_draw_boxes_color_yellow():
    processor = ImageProcessor()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [{"label": "defect", "confidence": 0.5, "bbox": [10, 10, 50, 50]}]
    out = processor._draw_boxes(dummy_img.copy(), detections)
    assert out.shape == dummy_img.shape

# 6. Проверка: метод get_color возвращает зелёный цвет для "good weld"
def test_get_color_green():
    from core.image_processor import ImageProcessor
    color = ImageProcessor.get_color("good weld", 0.9)
    assert color == (0, 255, 0)

# 7. Проверка: метод get_color возвращает красный цвет при высокой уверенности
def test_get_color_red():
    from core.image_processor import ImageProcessor
    color = ImageProcessor.get_color("defect", 0.95)
    assert color == (0, 0, 255)

# 8. Проверка: метод get_color возвращает жёлтый цвет при низкой уверенности
def test_get_color_yellow():
    from core.image_processor import ImageProcessor
    color = ImageProcessor.get_color("defect", 0.5)
    assert color == (0, 255, 255)

# 9. Проверка: если модель не находит боксов, возвращается пустой список
def test_no_boxes_detected(monkeypatch):
    class FakeYOLO:
        names = {0: "defect"}
        def __call__(self, image):
            class Result:
                boxes = []
            return [Result()]
    processor = ImageProcessor()
    processor.model = FakeYOLO()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated, detections = processor.detect_defects(img)
    assert detections == []

# 10. Проверка: повторный вызов load_model не перезагружает модель
def test_load_model_cached(monkeypatch):
    processor = ImageProcessor()
    class DummyModel:
        def __call__(self, img): return []
    processor.model = DummyModel()
    processor.load_model()  # не должно вызвать ошибку

# 11. Проверка: _draw_boxes корректно работает с несколькими объектами
def test_draw_boxes_multiple_detections():
    processor = ImageProcessor()
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = [
        {"label": "good weld", "confidence": 0.8, "bbox": [10, 10, 50, 50]},
        {"label": "crack", "confidence": 0.9, "bbox": [60, 60, 120, 120]},
    ]
    result = processor._draw_boxes(img.copy(), detections)
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape

# 12. Проверка: YOLO результат без .boxes не вызывает ошибку
def test_detect_defects_handles_missing_boxes(monkeypatch):
    class FakeYOLO:
        names = {0: "something"}
        def __call__(self, image):
            class Result:
                boxes = None
            return [Result()]
    processor = ImageProcessor()
    processor.model = FakeYOLO()
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated, detections = processor.detect_defects(dummy)
    assert detections == []

# 13. Проверка: пустое изображение возвращает пустой список детекций
def test_detect_defects_on_blank_image(monkeypatch):
    class FakeYOLO:
        names = {0: "anything"}
        def __call__(self, image):
            class Result: boxes = []
            return [Result()]
    processor = ImageProcessor()
    processor.model = FakeYOLO()
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    _, detections = processor.detect_defects(img)
    assert detections == []

# 14. Проверка: get_color нечувствителен к регистру
def test_get_color_case_insensitive():
    from core.image_processor import ImageProcessor
    color_upper = ImageProcessor.get_color("GOOD WELD", 0.8)
    color_lower = ImageProcessor.get_color("good weld", 0.8)
    assert color_upper == color_lower == (0, 255, 0)

# 15. Проверка: get_color для неизвестной метки и низкой уверенности
def test_get_color_unknown_label():
    from core.image_processor import ImageProcessor
    color = ImageProcessor.get_color("unknown", 0.3)
    assert color == (0, 255, 255)  # жёлтый, так как уверенность < 0.7
