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


def test_yolo_model_prediction():
    """Проверяет возможность обработки изображения моделью YOLO."""
    model = YOLO("models\YOLO_8.pt")
    results = model("tests/sample.jpg")
    result = results[0]
    assert hasattr(result, 'boxes'), "Результат обработки не содержит найденных объектов"

# ========== Тестирование интерфейса ==========

def test_clear_interface(app, qtbot):
    """Проверяет очистку интерфейса приложения."""
    app.loaded_image = np.zeros((100, 100, 3), dtype=np.uint8)
    app.annotated_image = np.zeros((100, 100, 3), dtype=np.uint8)
    app.textEdit.setPlainText("Какой-то текст")
    app.show_image_in_label(dummy_image, is_bgr=True)
    qtbot.wait(100)

    app.clear_interface()
    qtbot.wait(100)

    assert app.loaded_image is None, "loaded_image не очищен"
    assert app.annotated_image is None, "annotated_image не очищен"
    assert app.textEdit.toPlainText() == "", "textEdit не очищено"
    assert app.label_imagePreview.pixmap() is None, "label_imagePreview не очищен"


def test_export_report_without_text(app):
    """Проверяет, что нельзя экспортировать пустой отчет."""
    app.textEdit.clear()
    app.export_report()
    assert app.textEdit.toPlainText() == "", "Поле textEdit должно быть пустым после экспорта без текста"




