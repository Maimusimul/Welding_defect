import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QGroupBox, QMenuBar, QStatusBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class ManualUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Программа детектирования дефектов сварки")
        self.setGeometry(100, 100, 1000, 700)

        # Центральный виджет
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Заголовок
        self.label_header = QLabel(
            "<html><head/><body><p align='center'>"
            "<span style='font-size:18pt; font-weight:600;'>"
            "Программа детектирования дефектов сварки"
            "</span></p></body></html>"
        )
        self.label_header.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        main_layout.addWidget(self.label_header)

        # Верхний блок (изображение и информация)
        top_layout = QHBoxLayout()

        # Блок изображения
        self.label_imagePreview = QLabel("Здесь будет загруженное изображение")
        self.label_imagePreview.setAlignment(Qt.AlignCenter)
        self.label_imagePreview.setMinimumSize(400, 300)
        self.label_imagePreview.setStyleSheet("border: 1px solid gray")

        self.pushButton_5 = QPushButton("Загрузить фото")

        layout_image = QVBoxLayout()
        layout_image.addWidget(self.label_imagePreview)
        layout_image.addWidget(self.pushButton_5)

        groupBox_image = QGroupBox("Изображение")
        groupBox_image.setLayout(layout_image)

        # Блок информации
        self.label_filename = QLabel("Файл не выбран")
        self.label_resultInfo = QLabel()
        self.label_resultInfo.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label_resultInfo.setWordWrap(True)

        layout_info = QVBoxLayout()
        layout_info.addWidget(self.label_filename)
        layout_info.addWidget(self.label_resultInfo)

        groupBox_info = QGroupBox("Информация")
        groupBox_info.setLayout(layout_info)

        top_layout.addWidget(groupBox_image)
        top_layout.addWidget(groupBox_info)

        main_layout.addLayout(top_layout)

        # Текстовое поле
        self.textEdit = QTextEdit()
        layout_text = QVBoxLayout()
        layout_text.addWidget(self.textEdit)

        groupBox_textEdit = QGroupBox("Информация о дефектах")
        groupBox_textEdit.setLayout(layout_text)

        main_layout.addWidget(groupBox_textEdit)

        # Кнопки
        buttons_layout = QHBoxLayout()

        self.pushButton_2 = QPushButton("Сохранить результат")
        self.pushButton = QPushButton("Экспортировать отчет")
        self.pushButton_4 = QPushButton("Очистить интерфейс")
        self.pushButton_3 = QPushButton("Завершить работу")

        for btn in [self.pushButton_2, self.pushButton, self.pushButton_4, self.pushButton_3]:
            buttons_layout.addWidget(btn)

        main_layout.addLayout(buttons_layout)

        self.setCentralWidget(central_widget)

        # Меню и статусбар
        self.setMenuBar(QMenuBar(self))
        self.setStatusBar(QStatusBar(self))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ManualUI()
    window.show()
    sys.exit(app.exec_())
