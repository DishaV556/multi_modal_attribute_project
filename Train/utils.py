import cv2
import numpy as np

def preprocess_image(img, size=(64,64)):
    img = np.array(img)
    img = cv2.resize(img, size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img
