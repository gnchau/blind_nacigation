import tensorflow as tf
import numpy as np
import time
import cv2
from CyclicQueue import CyclicQueue
from utils import CLASSES, THRESH



model = tf.keras.models.load_model('../models/sidewalk_classifier_resnet.h5')

# start video
rotate = False
video = cv2.VideoCapture(0)
time.sleep(5)

if not video.read()[0]:
    vid = cv2.VideoCapture('/Users/gchau/Videos/tremont_st_01_21_2020.mp4')
    rotate = True
    position_video = 0.4
    video.set(cv2.CAP_PROP_POS_FRAMES, (position_video * vid.get(cv2.CAP_PROP_FRAME_COUNT)))

q = CyclicQueue(20, min_rat=0.8)
curr_class = ''

while True:
    # TODO: output vibrations to belt based on shift detector (opposite of shift offset)
    t1 = time.time()
    keyframe = cv2.resize(cv2.rotate(video.read()[1], cv2.ROTATE_90_COUNTERCLOCKWISE) if rotate else video.read()[1],
                          (480, 360))
    cap = cv2.resize(keyframe, (100, 100))
    raw_prediction = (model.predict(np.expand_dims(cap, 0))).tolist()[0]
    q.add(None if max(raw_prediction) < THRESH else raw_prediction)
    mean_pred = q.mean()
    if mean_pred is None:
        curr_class = 'Nothing detected'
    else:
        state = CLASSES[np.argmax(mean_pred)]
    t2 = time.time()
    confidence = max(mean_pred) * 100 // 1 if mean_pred is not None else None
    print('FPS:', 1/(t2 - t1), curr_class, 'Confidence:', confidence)
