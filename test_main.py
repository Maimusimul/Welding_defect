import pytest
import cv2
import numpy as np
dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
from ultralytics import YOLO
from main import MainWindow  # Импортируем твой основной код

# ========== Подготовка приложения для тестов интерфейса ==========

@pytest.fixture
def app(qtbot):
    """Фикстура для создания экземпляра приложения."""
    test_app = MainWindow()
    qtbot.addWidget(test_app)
    test_app.show()
    qtbot.waitExposed(test_app)
    return test_app

# ========== Юнит-тесты ==========

def test_load_image():
    """Проверяет возможность загрузки тестового изображения."""
    img = cv2.imread("tests/sample.jpg")
    assert img is not None, "Изображение не загружено"

# ========== Тестирование интерфейса ==========

def test_export_report_without_text(app):
    """Проверяет, что нельзя экспортировать пустой отчет."""
    app.textEdit.clear()
    app.export_report()
    assert app.textEdit.toPlainText() == "", "Поле textEdit должно быть пустым после экспорта без текста"





