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
        self.resize(1080, 1440)

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
        self.orig_img_path = None
        self.res_path = None

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.orig_img_path = file_path
            self.orig_img_pixmap = QPixmap(file_path).scaled(
                self.original_image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.original_image_label.setPixmap(self.orig_img_pixmap)

    def resizeEvent(self, event):
        if self.orig_img_path:
            self.orig_img_pixmap = QPixmap(self.orig_img_path).scaled(
                self.original_image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.original_image_label.setPixmap(self.orig_img_pixmap)

        if self.res_path:
            self.res_img_pixmap = QPixmap(self.res_path).scaled(
                self.processed_image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.processed_image_label.setPixmap(self.res_img_pixmap)

        super().resizeEvent(event)

    def process_image(self):
        if not self.orig_img_path:
            self.original_image_label.setText("Please upload an image first!")
            return

        # Get text input
        user_text = self.text_input.text()

        ############ Call the function to process the image and pass user text  ################
        self.communicate_with_yolo_world(self.orig_img_path, user_text)

        if self.res_path is not None:
            self.res_img_pixmap = QPixmap(self.res_path).scaled(
                self.processed_image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.processed_image_label.setPixmap(self.res_img_pixmap)

    def communicate_with_yolo_world(self, img_path: str, labels: str) -> None:
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

        self.res_path = f"{path_to_res}"

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
