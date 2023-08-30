from keras import utils
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, MaxPool2D, Activation, Dropout, Flatten
from keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import os
import pandas as pd
import plotly.graph_objs as go
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

import glob


def get_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for current_path, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(current_path, dr + "/*")))
    return count


train_dir = "citrus dataset/Train11/Train11"
test_dir = "citrus dataset/Test11"

train_samples = get_files(train_dir)
num_classes = len(glob.glob(train_dir + "/*"))
test_samples = get_files(test_dir)
print(num_classes, "Classes")
print(train_samples, "Train images")
print(test_samples, "Test images")

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
batch_size = 32
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_width, img_height), batch_size=batch_size)
test_generator = test_datagen.flow_from_directory(test_dir, shuffle=True, target_size=(img_width, img_height),
                                                  batch_size=batch_size)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu', name="conv2d_1"))
model.add(MaxPooling2D(pool_size=(3, 3), name="max_pooling2d_1"))
model.add(Conv2D(32, (3, 3), activation='relu', name="conv2d_2"))
model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2"))
model.add(Conv2D(64, (3, 3), activation='relu', name="conv2d_3"))
model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_3"))
model.add(Flatten(name="flatten_1"))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

from keras.callbacks import ReduceLROnPlateau

validation_generator = train_datagen.flow_from_directory(
    test_dir,  # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history_1 = model.fit(train_generator,
                      steps_per_epoch=None,
                      epochs=2, validation_data=validation_generator, validation_steps=None
                      , verbose=1,
                      callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.000001)],
                      use_multiprocessing=False,
                      shuffle=True)

model.evaluate(validation_generator)
model_json=model.to_json()
with open("model1.json","w") as json_file:
    json_file.write(model_json)
    model.save_weights("my_model_weight.h5")
    model.save('plantdiseasenaivecnn8epoch.h5')
    print("saved model to disk")
from keras.models import load_model



classes = list(train_generator.class_indices.keys())
import numpy as np
import matplotlib.pyplot as plt

# Pre-Processing test data same as train data.
img_width = 224
img_height = 224
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.preprocessing import image


def prepare(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255
    return np.expand_dims(x, axis=0)


result = model.predict([prepare(
    'Tomato___Early_blight/0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG')])

disease = image.load_img(
    'Tomato___Early_blight/0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG')
plt.imshow(disease)
print(result)

import numpy as np

classresult = np.argmax(result, axis=1)
print(classes[classresult[0]])
