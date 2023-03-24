import sys
import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
import pyk4a
from pyk4a import Config, PyK4A
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget


class PoseEstimation(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Load the reference pose image
        ref_image = cv2.imread('reference_pose.jpg')
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        # Set up the GUI layout
        self.ref_image_label = QLabel(self)
        self.ref_image_label.setPixmap(QPixmap.fromImage(
            QImage(ref_image, ref_image.shape[1], ref_image.shape[0], ref_image.strides[0], QImage.Format_RGB888)))
        self.ref_image_label.setFixedSize(640, 480)

        self.capture_label = QLabel(self)
        self.capture_label.setAlignment(Qt.AlignCenter)
        self.capture_label.setFixedSize(640, 480)

        self.text_label = QLabel(self)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setText("Press 'q' to quit.")
        self.text_label.setFixedSize(640, 30)

        hbox = QHBoxLayout()
        hbox.addWidget(self.ref_image_label)
        hbox.addWidget(self.capture_label)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.text_label)

        self.setLayout(vbox)
        self.setFixedSize(640, 540)

        # Set up the K4A device
        config = Config(color_resolution=pyk4a.ColorResolution.RES_720P)
        self.k4a = PyK4A(config)
        self.k4a.start()

        # Set up MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        # Set up the timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.show()

    def update_frame(self):
        # Capture a frame from the K4A device
        capture = self.k4a.get_capture()

        # Extract the color image
        color_image = capture.color

        # Convert the color image to RGB
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)

        # Use MediaPipe Pose to estimate the joint positions
        results = self.pose.process(color_image)

        # Draw annotations on the real-time image
        if results.pose_landmarks is not None:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                color_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Display the real-time image
        color_image = cv2.resize(color_image, (640, 480))
        color_image = QImage(color_image.data, color_image.shape[1], color_image.shape[0], color_image.strides[0], QImage.Format_RGB888)
        self.capture_label.setPixmap(QPixmap.fromImage(color_image))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()
            self.k4a.stop()
            self.k4a.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PoseEstimation()
    sys.exit(app.exec_())
