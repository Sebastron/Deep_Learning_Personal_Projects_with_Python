import os
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

dataset = []
label = []
INPUT_SIZE = 64

image_directory = "datasets/"
no_tumor_images = os.listdir(image_directory + "no/")
tumor_images = os.listdir(image_directory + "yes/")

for i, image_name in enumerate(no_tumor_images):
    if(image_name.split(".")[1] == "jpg"):
        image = cv2.imread(image_directory + "no/" + image_name)
        image = Image.fromarray(image, "RGB")
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(tumor_images):
    if(image_name.split(".")[1] == "jpg"):
        image = cv2.imread(image_directory + "yes/" + image_name)
        image = Image.fromarray(image, "RGB")
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

#If you want to generate two target variable as probability of two cases
#y_train = to_categorical(y_train, num_classes=2)
#y_test = to_categorical(y_test, num_classes=2)

#Create Artificial Neural Network
model = Sequential()

#First Layer
model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Hidden Layer N°1
model.add(Conv2D(32, (3,3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Hidden Layer N°2
model.add(Conv2D(32, (3,3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Last Layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1)) #Had only one Target Variable, Type Binary
model.add(Activation("sigmoid"))
#model.add(Dense(2)) #Had only one Target Variable, Type Binary
#model.add(Activation("softmax"))

# Loss Function
# Binary Cross Entropy = 1, sigmoid
# Categorical Cross Entropy = 2, softmax

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=20, validation_data=(x_test, y_test), shuffle=False)

model.save("models/BrainTumor20Epochs.h5")