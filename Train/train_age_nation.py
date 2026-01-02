import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

DATASET = "dataset/age_nationality"
X, y = [], []

for age in os.listdir(DATASET):
    for img in os.listdir(os.path.join(DATASET, age)):
        im = cv2.imread(os.path.join(DATASET, age, img))
        im = cv2.resize(im, (224,224))
        X.append(im)
        y.append(int(age))

X = np.array(X) / 255.0
y = np.array(y)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=12, batch_size=8)
model.save("models/age_model.h5")
