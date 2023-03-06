import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

INPUT_SIZE = 64
model = load_model('models/BrainTumor20Epochs.h5')
image = cv2.imread('datasets/pred/pred14.jpg')
img = Image.fromarray(image)
img = img.resize((INPUT_SIZE,INPUT_SIZE))
img = np.array(img)

#print(img)
input_img = np.expand_dims(img, axis=0)
result = model.predict(input_img)
print(result)