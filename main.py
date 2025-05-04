import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        uic.loadUi("1.ui", self)  # Загрузите нужный .ui файл

        # Изначально модель не загружаем
        self.model = None
        # Укажите ваш путь к 'best.pt'
        self.model_path = r"D:\4 курс 2 семестр\Defects\best.pt"

        # Подключаем сигналы к методам
        self.pushButton_5.clicked.connect(self.open_image)       # "Загрузить фото"
        self.pushButton_2.clicked.connect(self.save_results)     # "Сохранить результат"
        self.pushButton_3.clicked.connect(self.close_app)        # "Завершить работу"
        self.pushButton_4.clicked.connect(self.clear_interface)  # "Очистить интерфейс"
        self.pushButton.clicked.connect(self.export_report)      # "Экспортировать отчет"

        # Храним загруженное и аннотированное изображение (в формате BGR или RGB, в зависимости от этапа)
        self.loaded_image = None
        self.annotated_image = None

    def open_image(self):
        """Загружаем изображение, при необходимости — подгружаем модель YOLO и выводим результат."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать изображение",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_name:
            return

        self.label_filename.setText("Файл загружен")
        self.loaded_image = cv2.imread(file_name)  # BGR формат

        # Отобразим исходную картинку до детекции (по желанию)
        # Нужно конвертировать BGR -> RGB -> QPixmap
        self.show_image_in_label(self.loaded_image, is_bgr=True)

        # Если модель ещё не загружена — загружаем (лениво)
        if self.model is None:
            self.model = YOLO(self.model_path)

        # Прогоняем изображение через модель
        results = self.model(file_name)
        result = results[0]

        # Сформируем аннотированное изображение в памяти (уже в формате RGB)
        self.annotated_image = result.plot()

        # Формируем текст о найденных объектах
        text_info = []
        text_info.append(f"Всего найдено {len(result.boxes)} объектов.\n")
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]
            text_info.append(f"Класс: {class_name}, Уверенность: {conf:.2f}")

        # Выведем всё в textEdit
        self.textEdit.setText("\n".join(text_info))

        # И сразу же отобразим уже аннотированное изображение
        self.show_image_in_label(self.annotated_image, is_bgr=False)

    def show_image_in_label(self, image: np.ndarray, is_bgr: bool):
        """
        Преобразует numpy-массив (BGR или RGB) -> QImage -> QPixmap
        и показывает в label_imagePreview, сохраняя пропорции.
        """
        if image is None:
            return

        # Если картинка BGR, переводим её в RGB
        if is_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Создаём QImage из numpy-массива
        # image.shape: (height, width, channel)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qimage = QtGui.QImage(
            image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )

        # Создаём QPixmap из QImage
        pixmap = QtGui.QPixmap.fromImage(qimage)

        # Дополнительно масштабируем под размер label (с сохранением пропорций)
        label_width = self.label_imagePreview.width()
        label_height = self.label_imagePreview.height()
        scaled_pixmap = pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio)

        # Устанавливаем результат
        self.label_imagePreview.setPixmap(scaled_pixmap)

    def save_results(self):
        """
        Кнопка «Сохранить результат» сохраняет аннотированное изображение (self.annotated_image).
        """
        if self.annotated_image is None:
            self.textEdit.setText("Сначала загрузите фото и получите результат!")
            return

        # Открываем диалог для сохранения изображения
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить аннотированное изображение",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp);;Все файлы (*)",
            options=options
        )

        if file_name:
            # self.annotated_image картинка в RGB
            # Для сохранения через cv2.imwrite нужно сконвертировать в BGR
            img_bgr = cv2.cvtColor(self.annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_name, img_bgr)
            self.textEdit.append(f"Изображение сохранено: {file_name}")

    def export_report(self):
        """По-прежнему сохраняем текстовый отчёт, если вам нужно."""
        text = self.textEdit.toPlainText()
        if not text.strip():
            return

        report_content = f"Отчёт по детекции дефектов сварки\n\n{text}"
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Экспортировать отчет",
            "",
            "Текстовые файлы (*.txt);;Все файлы (*)",
            options=options
        )
        if file_name:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(report_content)

    def clear_interface(self):
        """Очищаем интерфейс."""
        self.label_filename.setText("Файл не выбран")
        self.textEdit.clear()
        self.loaded_image = None
        self.annotated_image = None
        # Очистить QLabel (убрать pixmap)
        self.label_imagePreview.clear()
        # Если нужно сбросить модель, раскомментируйте:
        # self.model = None

    def close_app(self):
        """Закрываем приложение."""
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
