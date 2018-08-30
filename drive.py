

from time import time
from PIL  import Image
from io   import BytesIO

import os
import cv2
import math
import numpy as np
import base64
import logging
from sklearn.externals import joblib
from keras.models import load_model

def logit(msg):
    print("%s" % msg)


class ImageProcessor(object):
    @staticmethod
    def show_image(img, name = "image", scale = 1.0, newsize=None):
        if scale and scale != 1.0:
            img = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC)

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, img)
        cv2.waitKey(1)


    @staticmethod
    def save_image(folder, img, prefix = "img", suffix = ""):
        from datetime import datetime
        filename = "%s-%s%s.jpg" % (prefix, datetime.now().strftime('%Y%m%d-%H%M%S-%f'), suffix)
        cv2.imwrite(os.path.join(folder, filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


    @staticmethod
    def rad2deg(radius):
        return radius / np.pi * 180.0


    @staticmethod
    def deg2rad(degree):
        return degree / 180.0 * np.pi


    @staticmethod
    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    @staticmethod
    def _normalize_brightness(img):
        maximum = img.max()
        if maximum == 0:
            return img
        adjustment = min(255.0/img.max(), 3.0)
        normalized = np.clip(img * adjustment, 0, 255)
        normalized = np.array(normalized, dtype=np.uint8)
        return normalized



class Car(object):
    MAX_STEERING_ANGLE = 40.0


    def __init__(self, control_function):
        self._driver = None
        self._control_function = control_function


    def drive(self, src_img):
        gray_image = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        model_steer = load_model('keras_cnn_model_steering_angle.h5')
        model_throttle = load_model('keras_cnn_model_throttle.h5')
        gray_image = cv2.resize(gray_image,(240,320))
        gray_image = np.array(gray_image,dtype=np.float32)
        img = np.reshape(gray_image,(-1,240,320,1))

        steer_angle = float(model_steer.predict(img, batch_size=1))
        throttle = model_throttle.predict(img, batch_size=1)
        print('predicted steer_angle={0}, throttle={1}'.format(steer_angle,throttle))
        self.control(steer_angle, throttle)


    def on_dashboard(self, dashboard):
        #normalize the units of all parameters
        last_steering_angle = np.pi/2 - float(dashboard["steering_angle"]) / 180.0 * np.pi
        throttle            = float(dashboard["throttle"])
        speed               = float(dashboard["speed"])
        img                 = ImageProcessor.bgr2rgb(np.asarray(Image.open(BytesIO(base64.b64decode(dashboard["image"])))))

        self.drive(img)


    def control(self, steering_angle, throttle):
        #convert the values with proper units
        #steering_angle = min(max(ImageProcessor.rad2deg(steering_angle), -Car.MAX_STEERING_ANGLE), Car.MAX_STEERING_ANGLE)
        self._control_function(steering_angle, throttle)


if __name__ == "__main__":
    import shutil
    import argparse
    from datetime import datetime

    import socketio
    import eventlet
    import eventlet.wsgi
    from flask import Flask

    parser = argparse.ArgumentParser(description='AutoDriveBot')
    parser.add_argument(
        'record',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder to record the images.'
    )
    args = parser.parse_args()

    if args.record:
        if not os.path.exists(args.record):
            os.makedirs(args.record)
        logit("Start recording images to %s..." % args.record)

    sio = socketio.Server()
    def send_control(steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            },
            skip_sid=True)

    car = Car(control_function = send_control)

    @sio.on('telemetry')
    def telemetry(sid, dashboard):
        if dashboard:
            car.on_dashboard(dashboard)
        else:
            sio.emit('manual', data={}, skip_sid=True)

    @sio.on('connect')
    def connect(sid, environ):
        car.control(0, 0)

    app = socketio.Middleware(sio, Flask(__name__))
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

# vim: set sw=4 ts=4 et