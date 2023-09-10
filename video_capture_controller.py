import cv2
import time
from CyclicQueue import CyclicQueue

try:
    stream = cv2.VideoCapture(0)
    cv2.resize(stream.read()[1], (100, 100))
except Exception as e:
    stream = cv2.VideoCapture('./data/turn_data.mp4')

queue = CyclicQueue(2)


def capturer():
    initial = True
    print("Video stream initialized.")
    while True:
        queue.add(stream.read()[1])

        # cann only capture so frequently
        if initial:
            time.sleep(20)
        else:
            time.sleep(0.1)
        initial = False


def get_images():
    return queue
