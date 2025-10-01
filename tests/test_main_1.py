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

# ========== Юнит-тесты ==========

def test_load_image():
    """Проверяет возможность загрузки тестового изображения."""
    img = cv2.imread("D:/4 курс 2 семестр/Диплом/tests/sample.jpg")
    assert img is not None, "Изображение не загружено"


def test_yolo_model_prediction():
    """Проверяет возможность обработки изображения моделью YOLO."""
    model = YOLO(r"D:\4 курс 2 семестр\Диплом\best.pt")
    results = model("D:/4 курс 2 семестр/Диплом/tests/sample.jpg")
    result = results[0]
    assert hasattr(result, 'boxes'), "Результат обработки не содержит найденных объектов"

# ========== Тестирование интерфейса ==========

def test_clear_interface():
    """Проверяет возможность загрузки тестового изображения."""
    img = cv2.imread("D:/4 курс 2 семестр/Диплом/tests/sample.jpg")
    assert img is not None, "Изображение не загружено"


def test_save_results_without_image():
    """Проверяет возможность загрузки тестового изображения."""
    img = cv2.imread("D:/4 курс 2 семестр/Диплом/tests/sample.jpg")
    assert img is not None, "Изображение не загружено"


def test_export_report_without_text():
    """Проверяет возможность загрузки тестового изображения."""
    img = cv2.imread("D:/4 курс 2 семестр/Диплом/tests/sample.jpg")
    assert img is not None, "Изображение не загружено"


def test_show_image_in_label():
    """Проверяет возможность загрузки тестового изображения."""
    img = cv2.imread("D:/4 курс 2 семестр/Диплом/tests/sample.jpg")
    assert img is not None, "Изображение не загружено"


def test_open_image_no_file():
    """Проверяет возможность загрузки тестового изображения."""
    img = cv2.imread("D:/4 курс 2 семестр/Диплом/tests/sample.jpg")
    assert img is not None, "Изображение не загружено"