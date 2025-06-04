import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QSize
from ultralytics import YOLO
import numpy as np


class MultiImageYOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Множественная обработка YOLO")
        self.setGeometry(100, 100, 1200, 600)

        self.model_path = "models/YOLO_11.pt"  # путь к модели YOLO
        self.model = None
        self.image_paths = []

        # ───────── UI ─────────
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Левая колонка: список изображений
        self.listWidget_images = QListWidget()
        self.listWidget_images.setIconSize(QSize(100, 100))
        self.listWidget_images.setViewMode(QListWidget.IconMode)
        self.listWidget_images.setResizeMode(QListWidget.Adjust)
        self.listWidget_images.setMinimumWidth(200)
        self.listWidget_images.itemClicked.connect(self.on_image_selected)

        # Кнопка загрузки
        self.button_load = QPushButton("Загрузить изображения")
        self.button_load.clicked.connect(self.open_images)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.button_load)
        left_layout.addWidget(self.listWidget_images)

        # Правая часть: отображение изображения
        self.image_label = QLabel("Выберите изображение")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 1px solid gray")

        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.image_label)

        self.setCentralWidget(main_widget)

    def open_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Выбрать изображения",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_names:
            return

        self.listWidget_images.clear()
        self.image_paths = file_names

        for file_path in file_names:
            item = QListWidgetItem()
            item.setText(os.path.basename(file_path))
            pixmap = QPixmap(file_path).scaled(100, 100, Qt.KeepAspectRatio)
            item.setIcon(QIcon(pixmap))
            item.setData(Qt.UserRole, file_path)
            self.listWidget_images.addItem(item)

    def on_image_selected(self, item):
        file_path = item.data(Qt.UserRole)
        if not os.path.exists(file_path):
            return

        image = cv2.imread(file_path)
        if image is None:
            return

        if self.model is None:
            self.model = YOLO(self.model_path)

        result = self.model(image)[0]
        annotated = result.plot()

        self.display_image(annotated)

    def display_image(self, image_bgr: np.ndarray):
        """Показать изображение в QLabel."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiImageYOLOApp()
    window.show()
    sys.exit(app.exec_())
