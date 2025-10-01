from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QScrollArea, QScrollBar
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint
import cv2


class NoScrollWheelArea(QScrollArea):
    def wheelEvent(self, event):
        event.ignore()  # Отключаем стандартный скролл QScrollArea


class ImageViewer(QDialog):
    def __init__(self, image_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Просмотр изображения")
        self.setFixedSize(1000, 800)

        self.zoom_factor = 1.0
        self.image_bgr = image_bgr

        # QLabel внутри scroll area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: white;")

        self.scroll_area = NoScrollWheelArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setStyleSheet("background-color: white;")

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)

        # Для перемещения
        self.drag_active = False
        self.last_mouse_pos = QPoint()

        self.update_image()

    def update_image(self):
        rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        new_w, new_h = int(w * self.zoom_factor), int(h * self.zoom_factor)

        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        qimg = QImage(resized.data, new_w, new_h, new_w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor * factor))
        self.update_image()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_active = True
            self.last_mouse_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.drag_active:
            delta = event.globalPos() - self.last_mouse_pos
            self.last_mouse_pos = event.globalPos()

            h_scroll: QScrollBar = self.scroll_area.horizontalScrollBar()
            v_scroll: QScrollBar = self.scroll_area.verticalScrollBar()

            h_scroll.setValue(h_scroll.value() - delta.x())
            v_scroll.setValue(v_scroll.value() - delta.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_active = False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
