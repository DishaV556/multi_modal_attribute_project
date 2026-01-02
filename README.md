🤖 Multi-Modal Attribute Prediction System


## 📌 Project Overview

The Multi-Modal Attribute Prediction System is a Streamlit-based AI application that performs multiple human and object attribute predictions using images, videos, audio, and real-time camera input.

This project integrates Computer Vision, Deep Learning, Audio Processing, and Rule-Based Logic into a single unified interface with 7 functional modules (tabs).

This project focuses on:

*Efficient document preprocessing

*Robust feature engineering

*Optimal model selection for different prediction tasks

*Integration of multiple predictive models into a single system

## 🎥 Project Preview

[![Watch the Preview](Project preview/tumbnail.png)]("https://github.com/USERNAME/REPO_NAME/raw/main/Project preview/Project preview.mp4")


## 🎯 Key Features

Age & Gender Detection from Images

Long Hair–Based Gender Logic

Senior Citizen Detection (Camera & Video)

Voice-Based Age & Emotion Detection

Sign Language (ASL) Alphabet Recognition

Car & People Detection with Car Color Analysis

Nationality-Aware Face Analysis

## 🧠 Technologies Used

Python

Streamlit

OpenCV

TensorFlow / Keras

YOLOv8 (Ultralytics)

NumPy, Pandas

PIL

Caffe Deep Learning Models

## 🏗️ System Architecture
...
User Input (Image / Video / Audio / Camera)
            ↓
Data Preprocessing
            ↓
Feature Engineering
            ↓
Model Inference
            ↓
Rule-Based / Conditional Logic
            ↓
Final Attribute Prediction
...

## 📂 Project Structure
multi_modal_ai_project/
│
├── .venu/
├── app.py
├── multi_modal_project.ipynb
├
├── dataset/
│    ├── age/
│    ├── age_nationality/
│    ├── car/
│    ├── emotion/
│    ├── emotion_nationality/
│    ├── hair_dataset/
│    ├── nationality/
│    ├── sign_language/
│    ├── test/
│    ├── voice/
│ 
├── models/
│   ├── face/
│   ├── age_gender_senior/
│   ├── age_gender/
│   ├── car/
│   ├── hair_model.keras
│   ├── voice/
│   ├── sign_language/
│   ├── nationality/
│   └── yolov8n.pt
│
├── utils/
│   └── audio_features.py
│   └── _init_.py
│
├── utlis/
│   └── dress_color.py
│   └── face_utils.py
│   └── image_preprocess.py
│
├── data/
│   └── senior_citizen/
│       └── senior_citizens.csv
│   └──age_gender\
│   └──long_hair prediction\
│   └──nationality\
│   └──senior_citizen\
│   └──sign_language\
│   └──voice\
│   └──voice\
│
│── haarcascade_frontalface_default.xml
├── temp.wav
├── requriement.txt
└── README.md

## 🧹 Data Preprocessing (Used Across Modules)
🔹 Image Preprocessing

Convert to NumPy array

Resize to model-specific input size (64×64, 224×224, 227×227)

Normalize pixel values (/255.0)

Expand dimensions for batch input

🔹 Audio Preprocessing

WAV file input

MFCC feature extraction using extract_mfcc

Reshape features for ML models

🔹 Video Preprocessing

Frame-by-frame extraction using OpenCV

Temporary file handling using tempfile

Real-time frame processing

## ⚙️ Feature Engineering

🖼️ Image Features

Face region extraction using Haar Cascade / DNN

CNN-based feature extraction

HSV color space analysis (car color detection)

Pixel intensity averaging (dress color detection)

🔊 Audio Features

MFCC coefficients

Temporal audio features flattened into vectors

🎥 Video Features

Object bounding boxes (YOLO)

Frame-level attribute aggregation

## 🧠 Model Selection & Logic

🔹 Models Used

Module	Model
Age & Gender	Caffe DNN
Hair Detection	CNN (Keras .keras)
Senior Citizen	Caffe DNN
Voice Analysis	Keras (.h5)
Sign Language	CNN
Car Detection	YOLOv8
Nationality	CNN

## 🧩 Application Modules (Tabs)

🧑 Tab 1: Age & Gender Detection

Face detection using OpenCV DNN

Age classification into predefined buckets

Gender classification (Male/Female)

Output:
Age group + Gender + annotated image

💇 Tab 2: Long Hair Prediction with Gender Logic

CNN predicts Long / Short Hair

Random ML-based age & gender fallback

Rule Applied:

Age 20–30 → Hair overrides gender

Else → ML gender result

🧓 Tab 3: Senior Citizen Detection

Camera or video input

Face detection + age estimation

If age ≥ 40 → flagged as senior citizen

Logs results to CSV with timestamp

🎧 Tab 4: Voice-Based Age & Emotion Detection

MFCC feature extraction

Gender validation

Age prediction

Emotion detection for senior citizens

🧏 Tab 5: Sign Language Detection (ASL)

CNN-based alphabet classification (A–Z)

Image upload + camera support

Time-restricted access (6 PM – 10 PM)

🚦 Tab 6: Car Color & People Detection

YOLOv8 object detection

Car vs Person classification

HSV-based blue car detection

Supports image and video input

🌍 Tab 7: Nationality-Aware Face Analysis

Nationality prediction

Emotion recognition

Conditional predictions:

Indian → Age + Dress Color

US → Age

African → Dress Color

## 📊 Output & Visualization

Bounding boxes and labels

Confidence scores

CSV logging for senior citizens

Real-time Streamlit UI updates

▶️ How to Run
pip install -r requirements.txt
streamlit run app.py

🚀 Future Enhancements

Cloud deployment (AWS / GCP)

API support for mobile integration

## 🔗 GitHub & Other Accounts
 🔗 Connect with Me

- **GitHub:** https://github.com/DishaV556?tab=repositories
- **LinkedIn:** https://www.linkedin.com/in/disha-vishwakarma-985b75286/

## 👩‍💻 Author

Disha Vishwakarma
IT Engineering 
AI / ML | Data Science | Computer Vision
Multi-Modal Intelligent Systems Developer
