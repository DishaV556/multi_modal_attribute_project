import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from utils.audio_features import extract_mfcc

X, y = [], []
base_path = "dataset/age"

for age_folder in os.listdir(base_path):
    folder = os.path.join(base_path, age_folder)
    age_value = int(age_folder.replace("_plus", ""))

    for file in os.listdir(folder):
        X.append(extract_mfcc(os.path.join(folder, file)))
        y.append(age_value)

X = np.array(X)
y = np.array(y)

model = Sequential([
    Dense(128, activation='relu', input_shape=(40,)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=40, batch_size=16)

model.save("models/age_model.h5")
print("✅ Age model saved")
