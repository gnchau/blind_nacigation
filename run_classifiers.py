import threading
from shift_direction_classifier.shift_direction_classifier import SidewalkClassifier as SidewalkClassifier
from rotation_classifier.rotation_classifier import RotationClassifier as RotationClassifier
import video_capture_controller
import time

threading.Thread(target=video_capture_controller.capturer).start()
time.sleep(1)

shift_classifier = SidewalkClassifier()
rotation_classifier = RotationClassifier()

cnt = 0
while True:
    vid = video_capture_controller.get_images().get_last()

    shift_classifier_pred = shift_classifier.get_prediction()
    rotation_classifier_pred = rotation_classifier.get_prediction()
    print(shift_classifier_pred)
    print(rotation_classifier_pred)
    cnt += 1
    print(f'Evaluated {cnt} images.')
