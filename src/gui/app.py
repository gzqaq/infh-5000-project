import sys

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QFrame, QGridLayout,
                             QLabel, QPushButton, QTextEdit, QVBoxLayout,
                             QWidget)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("A Fun App")
        self.setGeometry(100, 100, 800, 600)

        # Font settings
        font = QFont("Microsoft YaHei", 12)  # Font: Microsoft YaHei, Size: 12
        self.setFont(font)

        # Main Layout
        main_layout = QGridLayout()

        # Widgets
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setFont(font)
        self.upload_button.clicked.connect(self.upload_image)

        self.submit_button = QPushButton("Submit")
        self.submit_button.setFont(font)
        self.submit_button.setFixedWidth(
            self.upload_button.sizeHint().width()
        )  # Ensure width is consistent
        self.submit_button.clicked.connect(self.process_image)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter some text here...")
        self.text_input.setFixedWidth(self.upload_button.sizeHint().width() * 1.2)
        self.text_input.setFont(font)

        # Frames for displaying images with square size
        self.original_frame = QFrame()
        self.original_frame.setFrameShape(QFrame.Box)
        self.original_frame.setFrameShadow(QFrame.Sunken)
        self.original_frame.setStyleSheet("background-color: lightgray;")
        self.original_frame.setFixedSize(300, 300)  # Predefined square size
        self.original_image_label = QLabel("Original Image")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setFont(font)
        self.original_frame.setLayout(QVBoxLayout())
        self.original_frame.layout().addWidget(self.original_image_label)

        self.processed_frame = QFrame()
        self.processed_frame.setFrameShape(QFrame.Box)
        self.processed_frame.setFrameShadow(QFrame.Sunken)
        self.processed_frame.setStyleSheet("background-color: lightgray;")
        self.processed_frame.setFixedSize(300, 300)  # Predefined square size
        self.processed_image_label = QLabel("Processed Image")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setFont(font)
        self.processed_frame.setLayout(QVBoxLayout())
        self.processed_frame.layout().addWidget(self.processed_image_label)

        # Add widgets to layout
        main_layout.addWidget(
            self.upload_button, 0, 0, 1, 1, Qt.AlignLeft | Qt.AlignTop
        )
        main_layout.addWidget(self.submit_button, 1, 0, 1, 1, Qt.AlignLeft)
        main_layout.addWidget(self.text_input, 2, 0, 1, 1, Qt.AlignLeft)

        main_layout.addWidget(self.original_frame, 0, 1, 3, 1, Qt.AlignCenter)
        main_layout.addWidget(self.processed_frame, 0, 2, 3, 1, Qt.AlignCenter)

        # Set layout
        self.setLayout(main_layout)

        # Variables
        self.image_path = None

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.original_image_label.setPixmap(
                pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def process_image(self):
        if not self.image_path:
            self.original_image_label.setText("Please upload an image first!")
            return

        # Get text input
        user_text = self.text_input.toPlainText()

        ############ Call the function to process the image and pass user text  ################
        processed_image = self.convert_to_black_and_white(self.image_path, user_text)

        # Convert the processed image to QPixmap and display it
        height, width = processed_image.shape
        bytes_per_line = width
        q_image = QImage(
            processed_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_Grayscale8,
        )
        pixmap = QPixmap.fromImage(q_image)
        self.processed_image_label.setPixmap(
            pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def convert_to_black_and_white(self, image_path, user_text):
        # Process the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # For demonstration, print the input text (you can implement custom logic using `user_text`)
        print(f"User text: {user_text}")

        return image


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
