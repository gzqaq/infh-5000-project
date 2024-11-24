import json
import os
import sys
import time
from pathlib import Path

import cv2
from PyQt5.QtCore import QLibraryInfo

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QVBoxLayout,
                             QWidget)


class App(QWidget):
    def __init__(self, msg_file: Path):
        super().__init__()
        self._init_ui()

        self.msg_file = msg_file

    def _init_ui(self):
        self.setWindowTitle("A Fun App")
        self.resize(800, 600)

        # Widgets
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)

        self.submit_button = QPushButton("Submit")
        self.submit_button.setFixedWidth(
            self.upload_button.sizeHint().width()
        )  # Ensure width is consistent
        self.submit_button.clicked.connect(self.process_image)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter labels to detect...")

        # Frames for displaying images with square size
        self.original_frame = QFrame()
        self.original_frame.setFrameShape(QFrame.Box)
        self.original_image_label = QLabel("Original Image")
        self.original_image_label.setAlignment(Qt.AlignCenter)

        self.processed_frame = QFrame()
        self.processed_frame.setFrameShape(QFrame.Box)
        self.processed_image_label = QLabel("Processed Image")
        self.processed_image_label.setAlignment(Qt.AlignCenter)

        # top layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.upload_button)
        top_layout.addWidget(self.submit_button)
        top_layout.addWidget(self.text_input)

        # image frames
        image_layout = QHBoxLayout()
        self.original_frame.setLayout(QVBoxLayout())
        self.original_frame.layout().addWidget(self.original_image_label)
        self.processed_frame.setLayout(QVBoxLayout())
        self.processed_frame.layout().addWidget(self.processed_image_label)

        # main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.original_frame)
        main_layout.addWidget(self.processed_frame)

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
        user_text = self.text_input.text()

        ############ Call the function to process the image and pass user text  ################
        # processed_image = self.convert_to_black_and_white(self.image_path, user_text)
        processed_image = self.communicate_with_yolo_world(self.image_path, user_text)

        # Convert the processed image to QPixmap and display it
        height, width, channels = processed_image.shape
        bytes_per_line = width * channels
        q_image = QImage(
            processed_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(q_image)
        self.processed_image_label.setPixmap(
            pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def communicate_with_yolo_world(self, img_path: str, labels: str) -> np.ndarray:
        path_to_orig = Path(img_path).resolve()
        path_to_res = path_to_orig.with_suffix(f".res{path_to_orig.suffix}")

        if not path_to_res.exists():
            path_to_res.touch()
        timestamp = path_to_res.stat().st_mtime_ns

        msg = {
            "img_path": f"{path_to_orig}",
            "labels": labels,
            "tgt_path": f"{path_to_res}",
        }
        with open(self.msg_file, "w") as fd:
            json.dump(msg, fd)

        while path_to_res.stat().st_mtime_ns == timestamp:
            time.sleep(0.1)

        img_bgr = cv2.imread(path_to_res)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def convert_to_black_and_white(self, image_path, user_text):
        # Process the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # For demonstration, print the input text (you can implement custom logic using `user_text`)
        print(f"User text: {user_text}")

        return image


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App(Path(sys.argv[1]).resolve())
    window.show()
    sys.exit(app.exec_())
