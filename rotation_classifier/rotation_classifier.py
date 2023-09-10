import tensorflow as tf
import numpy as np
import cv2
import threading
import video_capture_controller
from utils import CLASSES, PREPROCESSING_DIMS, BUFFER_SIZE, DET_THRESH
from CyclicQueue import CyclicQueue

labels = ['Left Turn', 'No Turn', 'Right Turn']
model_path = "../models/rotation_classification_model_RESNET.h5"


class RotationClassifier:
    """
    Performs rotation classification. TODO: consolidate rotation and translation classifiers.
    """
    def __init__(self):
        self.model = tf.keras.models.load_model(model_path)
        self.curr_buffer = CyclicQueue(BUFFER_SIZE, min_rat=0.5)
        self.image_queue = CyclicQueue(1)
        self.classifier_queue = CyclicQueue(1)
        threading.Thread(target=self.init_classifier).start()

    def capture(self):
        while True:
            try:
                frame = video_capture_controller.get_images().get_last()
                if frame:
                    preprocessed_frame = cv2.resize(frame, PREPROCESSING_DIMS,
                                                    interpolation=cv2.INTER_LINEAR)
                    self.image_queue.add(np.expand_dims(preprocessed_frame, 0))
            except Exception as e:
                print(e)

    def predict(self, img):
        model_output = self.model.predict(img).tolist()[0]
        self.curr_buffer.add(None if not model_output or max(model_output) < DET_THRESH else model_output)
        avg_res = self.curr_buffer.mean()
        self.classifier_queue.add(CLASSES[len(CLASSES) - 1] if not avg_res else CLASSES[np.argmax(avg_res)])

    def init_classifier(self):
        # multithread classifier
        threading.Thread(target=self.capture).start()
        while True:
            try:
                self.predict(self.image_queue.get_last())
            except Exception as e:
                print(e)

    def get_prediction(self):
        return self.classifier_queue.get_last()
