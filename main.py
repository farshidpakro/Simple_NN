import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.utils import to_categorical
from keras import models
from PIL import Image
# import seaborn as sns
import random
import itertools
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os

if len(sys.argv) < 2:
    print("Usage: main.py <picture_file>")
    sys.exit(1)

picture_file = sys.argv[1]
# picture_file = "C:\\Users\\farshid\\Desktop\\one.png"


# Load the picture
image = cv2.imread(picture_file, cv2.IMREAD_GRAYSCALE)

# Resize the image to 28x28
resized_image = cv2.resize(image, (28, 28))

# Reshape the image to add a single channel dimension
reshaped_image = resized_image.reshape(1, 28, 28, 1)

# Normalize the image data
normalized_image = reshaped_image / 255.0





# data_dir = os.path.dirname(__file__)

# x_train = pd.read_csv(os.path.join(data_dir, 'mnist_train.csv'))

# # x_train =  pd.read_csv('mnist_train.csv')
# x_test = pd.read_csv(os.path.join(data_dir, 'mnist_test.csv'))
# # x_test =  pd.read_csv("mnist_test.csv")
# y_train = x_train.iloc[:, 0]
# y_test = x_test.iloc[:, 0]
# x_train = x_train.drop(x_train.columns[0], axis=1)
# x_test = x_test.drop(x_test.columns[0], axis=1)
# train_images = x_train
# train_labels = y_train
# test_images = x_test
# test_labels = y_test

# train_images =train_images.values.reshape(59999,28,28,1).astype('float32') / 255
# test_images =test_images.values.reshape(9999,28,28,1).astype('float32') / 255
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)

# # Create a simple CNN model
# model = models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(train_images, y_train, epochs=10, validation_data=(test_images, y_test))
# model.evaluate(train_images, y_train)

# load model
model = tf.keras.models.load_model('model.h5')

# Use the model to predict the image
# predictions = model.predict(img_array.reshape(-1,28,28,1).astype('float32')/255)
predictions = model.predict(normalized_image)
# Get the predicted number
predicted_numbers = np.argmax(predictions, axis=1)
# print ( y_test.to_numpy())

print("The predicted numbers in the images are:", predicted_numbers)