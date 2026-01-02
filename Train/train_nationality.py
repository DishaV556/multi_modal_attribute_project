import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

DATASET = "dataset/nationality"
CLASSES = ["Indian", "United_States", "African", "Other"]

X, y = [], []

for idx, label in enumerate(CLASSES):
    path = os.path.join(DATASET, label)

    if not os.path.exists(path):
        print(f"⚠️ Folder missing, skipping: {path}")
        continue

    images = os.listdir(path)

    if len(images) == 0:
        print(f"⚠️ No images found in: {path}")
        continue

    for img_name in images:
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (224, 224))
        X.append(img)
        y.append(idx)

for idx, label in enumerate(CLASSES):
    path = os.path.join(DATASET, label)
    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name))
        img = cv2.resize(img, (224,224))
        X.append(img)
        y.append(idx)

X = np.array(X) / 255.0
y = to_categorical(y)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=8)
model.save("models/nationality_model.h5")
