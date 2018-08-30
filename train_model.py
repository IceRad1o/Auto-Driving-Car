#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda, Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import print_summary
from keras import backend as K
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import ntpath
from sklearn.externals import joblib

__author__ = 'Bin Yin'
__date__ = '2018-08'
__version_info__ = (0, 0, 0)
__version__ = '.'.join(str(i) for i in __version_info__)
# virtual env : car-behavioral-cloning

LOG_ROOT = 'Log'


def get_training_dataset():
    # x
    image_path_list = []
    image_gray_array = []
    speed_list = []
    # y
    steering_angle_list = []
    throttle_list = []
    
    df = pd.read_csv(os.path.join(LOG_ROOT, 'driving_log.csv'), header=None, usecols=[0,1,2,3,4,5,6], names=['driving_snapshot_image_path', \
            'steering_angle', 'throttle', 'brake', 'speed', 'timestamp', 'lap'])
    for index, row in df.iterrows():
        # x: image path
        path_origin = row['driving_snapshot_image_path']
        image_file_name = ntpath.basename(path_origin)
        path_real = os.path.join( os.path.join(LOG_ROOT, 'IMG'), image_file_name )
        image_path_list.append(path_real)
        # x: image array
        img = cv2.imread(path_real, cv2.IMREAD_GRAYSCALE)# IMREAD_GRAYSCALE, IMREAD_COLOR
        image_gray_array.append(img)
        # x: speed
        speed = float(row['speed'])
        speed_list.append(speed)
        
        # y: steering_angle (+45 degrees to -45 degrees.)
        steering_angle = float(row['steering_angle'])
        steering_angle_list.append(steering_angle)
        # y: throttle (The current throttle value. Ranges from 0 to 1)
        throttle = float(row['throttle'])
        throttle_list.append(throttle)
    
    dataset = {
        'x_img_arr': np.array(image_gray_array),
        'y_steering_angle': np.array(steering_angle_list),
        'y_throttle': np.array(throttle_list),
        
    }
    return dataset


def train_cnn_model(x_data, y_data, mname):
    # data preprocess
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    batch_size = 28
    epochs = 100

    # input image dimensions
    img_rows, img_cols = x_data.shape[1], x_data.shape[2]
    if K.image_dim_ordering() == 'th':#K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    # create model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())
    
    # call back for tensorboard
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph_CNN_{0}'.format(mname), 
                                             write_graph=True, 
                                             write_images=True, 
                                             embeddings_layer_names=None, 
                                             embeddings_metadata=None)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tbCallBack])
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('cnn_model_{0}.h5'.format(mname))

def keras_model(x_data,y_data,mname):
    #data process
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    batch_size = 28
    epochs = 100

    # input image dimensions
    img_rows, img_cols = x_data.shape[1], x_data.shape[2]
    if K.image_dim_ordering() == 'th':  # K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # model create
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(128))

    model.add(Dense(64))
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=0.0001), loss="mse")
    filepath = 'keras_cnn_model_{0}.h5'.format(mname)
    checkpoint1 = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint1]

    model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=5, batch_size=64, callbacks=callbacks_list)
    print_summary(model)
    model.save('keras_cnn_model_{0}.h5'.format(mname))



def train_rf_model(x_data, y_data, mname):
    print('start train rf model {0}'.format(mname))
    start_time = time.time()

    # data preprocess
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1]*x_data.shape[2]))
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = RandomForestRegressor( n_estimators=500,n_jobs=4 )
    model.fit(x_train,y_train)
    score_test = model.score(x_test,y_test)
    print(score_test)#The best possible score is 1.0
    end_time = time.time()
    print('time cost : {0}s'.format(end_time - start_time))
    joblib.dump(model, 'rf_model_{0}.jl'.format(mname))


def main():
    dataset = get_training_dataset()
    x1_data = dataset['x_img_arr']
    y1_data = dataset['y_steering_angle']
    x2_data = dataset['x_img_arr']
    y2_data = dataset['y_throttle']

    keras_model(x1_data,y1_data, 'steering_angle')
    keras_model(x2_data,y2_data, 'throttle')


if __name__=='__main__':
    main()



