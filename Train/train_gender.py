import os
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from utils.audio_features import extract_mfcc

X, y = [], []
base_path = "dataset/gender"

labels = {"male": 0, "female": 1}

for gender in labels:
    folder = os.path.join(base_path, gender)
    for file in os.listdir(folder):
        X.append(extract_mfcc(os.path.join(folder, file)))
        y.append(labels[gender])

X = np.array(X)
y = to_categorical(y)

model = Sequential([
    Dense(128, activation='relu', input_shape=(40,)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=30, batch_size=16)

model.save("models/gender_model.h5")
print("✅ Gender model saved")
