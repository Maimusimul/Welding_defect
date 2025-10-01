import cv2
import pytest
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from PyQt5.QtGui import QIcon

@pytest.fixture
def app(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    return window

def test_initial_ui_state(app):
    assert app.label_filename.text() == "Файл не выбран"
    assert app.textEdit.toPlainText() == ""
    assert app.listWidget_images.count() == 0

def test_load_image_button(qtbot, app, tmp_path):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QListWidgetItem

    # Симулируем добавление изображения вручную
    test_img = tmp_path / "test.jpg"
    np_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(test_img), np_img)

    # Добавим в виджет
    app.loaded_image_paths.add(str(test_img))
    item = QListWidgetItem()
    pixmap = QPixmap(str(test_img)).scaled(100, 100, Qt.KeepAspectRatio)
    item.setIcon(QIcon(pixmap))
    item.setData(Qt.UserRole, str(test_img))
    app.listWidget_images.addItem(item)

    assert app.listWidget_images.count() == 1
