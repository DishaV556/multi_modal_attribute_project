import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from utils.audio_features import extract_mfcc

X, y = [], []
base_path = "dataset/emotion"

emotions = {"happy": 0, "sad": 1, "angry": 2, "neutral": 3}

for emotion in emotions:
    folder = os.path.join(base_path, emotion)
    for file in os.listdir(folder):
        X.append(extract_mfcc(os.path.join(folder, file)))
        y.append(emotions[emotion])

X = np.array(X)
y = to_categorical(y)

model = Sequential([
    Dense(128, activation='relu', input_shape=(40,)),
    Dense(64, activation='relu'),
    Dense(len(emotions), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=35, batch_size=16)

model.save("models/emotion_model.h5")
print("✅ Emotion model saved")
