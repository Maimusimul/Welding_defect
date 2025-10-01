import pytest
import cv2
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

