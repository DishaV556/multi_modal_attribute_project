import streamlit as st                     #if there is an error in import download it or delete existing venu and paste this code in terminal
import cv2                                     #py -3.10 -m venv venv
import numpy as np                            #venv\Scripts\activate
from PIL import Image                           #python -m pip install --upgrade pip
from tensorflow.keras.models import load_model
import pandas as pd                     #pip install tensorflow streamlit opencv-python ultralytics numpy pillow pandas
from datetime import datetime
import os
import tempfile
from utils.audio_features import extract_mfcc
from ultralytics import YOLO

def preprocess_image(img, size=(224,224)):
    img = np.array(img)
    img = cv2.resize(img, size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img



hair_model = load_model("models/hair_model.keras")



def predict_age_gender(img):
    import random
    age = random.randint(15, 45)
    gender = random.choice(["Male", "Female"])
    return age, gender


# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="  Multi-Modal Detection", layout="centered")
st.title("🤖 Multi-Modal Attribute Detection")
st.write ("Click ➺ arrow button to go to next tabs")


# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4,tab5,tab6,tab7 = st.tabs([
    "🧑 Age & Gender",
    "💇 Long hair prediction",
    "🧓 Senior citizen ",
    "🎧 Age and Emotion Detection from Voice",
     "🧏 Sign Language Detection",
     "🚦 Car-colour detection Model",
    "🌍 Nationality Detection Model"
])


# ------------------ LOAD MODELS ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_PROTO = os.path.join(
    BASE_DIR, "models", "face", "deploy.prototxt"
)

FACE_MODEL = os.path.join(
    BASE_DIR, "models", "face",
    "res10_300x300_ssd_iter_140000.caffemodel"
)
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)



AGE_PROTO = os.path.join(
    BASE_DIR, "models", "age_gender", "age_deploy.prototxt"
)

AGE_MODEL = os.path.join(
    BASE_DIR, "models", "age_gender", "age_net.caffemodel"
)

GENDER_PROTO = os.path.join(
    BASE_DIR, "models", "age_gender", "gender_deploy.prototxt"
)

GENDER_MODEL = os.path.join(
    BASE_DIR, "models", "age_gender", "gender_net.caffemodel"
)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)



# ------------------ Constants ------------------
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# ================= TAB 1 ==========================
with tab1:
    st.subheader("🧑 Age & Gender Detection")
    st.write("Upload an image and click **Analyze** to detect age and gender.")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="age_gender_tab1")

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        img_array = np.array(image)
        st.image(image, caption="Uploaded Image", width=400)  # Using width instead of deprecated use_column_width

        analyze_btn = st.button("Analyze")

        if analyze_btn:
            frame = img_array.copy()
            h, w = frame.shape[:2]

            # -------- Face Detection --------
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))
            face_net.setInput(blob)
            detections = face_net.forward()

            if detections.shape[2] == 0:
                st.error("No face detected!")
            else:
                face_found = False
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.7:
                        face_found = True
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)

                        pad = 20
                        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

                        face = frame[y1:y2, x1:x2]
                        if face.size == 0:
                            continue

                        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                        # -------- Gender --------
                        gender_net.setInput(face_blob)
                        gender = GENDER_LIST[gender_net.forward()[0].argmax()]

                        # -------- Age --------
                        age_net.setInput(face_blob)
                        age = AGE_LIST[age_net.forward()[0].argmax()]

                        # Draw rectangle + label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{gender}, {age}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        st.success(f"Detected Gender: **{gender}**")
                        st.success(f"Detected Age Group: **{age}**")

                        break  # Only detect the first face

                if not face_found:
                    st.warning("No face detected. Please upload a clear image.")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Result", width=500)
        else:
            st.info("👆 Click **Analyze** to start detection.")

# ================= TAB 2 ==========================
with tab2:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    HAIR_MODEL_PATH = os.path.join(BASE_DIR, "models", "hair_length_model.h5")

    # ================= LOAD MODEL =================
    hair_model = load_model(HAIR_MODEL_PATH)


    # ================= PREPROCESS =================
    def preprocess_hair(img, size=(224, 224)):
        img = cv2.resize(img, size)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        return img


    # ================= SIMPLE GENDER HEURISTIC =================
    def estimate_gender_basic(img):
        """
        Lightweight fallback gender estimation
        Used ONLY outside 20–30 age range
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # Very light heuristic (safe)
        if brightness > 140:
            return "Female"
        else:
            return "Male"


    # ================= UI HEADER =================
    st.markdown("""
    Age-aware gender logic with ML + rule-based reasoning
    </p>
    """, unsafe_allow_html=True)

    # ================= TASK INFO =================
    with st.expander("📌 Task Rules (Click to View)"):
        st.markdown("""
    - **Age 20–30**
      - Long Hair → **Female**
      - Short Hair → **Male**
    - **Outside 20–30**
      - Gender predicted normally (hair ignored)
    - Focus is on **logic + GUI**
    """)

    # ================= INPUT =================
    uploaded_file = st.file_uploader(
        "📤 Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    age_input = st.slider(
        "🎂 Select estimated age",
        min_value=10,
        max_value=60,
        value=25
    )

    analyze = st.button("🔍 Analyze", use_container_width=True)

    # ================= PROCESS =================
    if uploaded_file and analyze:
        with st.spinner("Analyzing image..."):
            image = Image.open(uploaded_file).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # ---- Hair Prediction ----
            hair_input = preprocess_hair(frame)
            hair_score = float(hair_model.predict(hair_input)[0][0])

            if hair_score >= 0.6:
                hair_type = "Long"
            elif hair_score <= 0.4:
                hair_type = "Short"
            else:
                hair_type = "Uncertain"

            # ---- Gender (Fallback) ----
            base_gender = estimate_gender_basic(frame)

            # ---- TASK LOGIC ----
            if 20 <= age_input <= 30:
                if hair_type == "Long":
                    final_gender = "Female"
                elif hair_type == "Short":
                    final_gender = "Male"
                else:
                    final_gender = base_gender

                logic_used = "Age 20–30 → Hair-based override applied"
            else:
                final_gender = base_gender
                logic_used = "Age outside 20–30 → Normal gender estimation"

        # ================= DISPLAY =================
        st.divider()
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.subheader("📊 Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Selected Age", age_input)
            st.metric("Hair Score", round(hair_score, 2))
            st.metric("Hair Type", hair_type)

        with col2:
            st.metric("Estimated Gender", base_gender)
            st.success(f"Final Gender: {final_gender}")

        st.info(f"🧠 Decision Logic: {logic_used}")

    elif uploaded_file and not analyze:
        st.info("👆 Click **Analyze** to run prediction")
# ================= TAB 3 ==========================


# ================= SESSION STATE =================
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
# ================= LOAD MODELS =================

FACE_CASCADE = "haarcascade_frontalface_default.xml"
AGE_PROTO = "models/age_gender_senior/age_deploy1.prototxt"
AGE_MODEL = "models/age_gender_senior/age_net1.caffemodel"
GENDER_PROTO = "models/age_gender_senior/gender_deploy1.prototxt"
GENDER_MODEL = "models/age_gender_senior/gender_net1.caffemodel"

assert os.path.exists(AGE_PROTO), f"Missing file: {AGE_PROTO}"
assert os.path.exists(AGE_MODEL), f"Missing file: {AGE_MODEL}"
assert os.path.exists(GENDER_PROTO), f"Missing file: {GENDER_PROTO}"
assert os.path.exists(GENDER_MODEL), f"Missing file: {GENDER_MODEL}"

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)


AGE_BUCKETS = [
    '(0-2)', '(4-6)', '(8-12)', '(15-20)',
    '(25-32)', '(38-43)', '(48-53)', '(60-100)'
]
GENDERS = ['Male', 'Female']

CSV_FILE = "data/senior_citizen/senior_citizens.csv"

# ================= INIT CSV =================
with tab3:
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=["Age", "Gender", "Time"]).to_csv(CSV_FILE, index=False)

    # ================= SESSION STATE =================
    if "detections" not in st.session_state:
        st.session_state.detections = []

    if "video_done" not in st.session_state:
        st.session_state.video_done = False


    # ================= HELPERS =================
    def get_age_value(bucket):
        low, high = bucket[1:-1].split('-')
        return (int(low) + int(high)) // 2


    def process_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]

            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.426, 87.769, 114.897),
                swapRB=False
            )

            # Gender
            gender_net.setInput(blob)
            gender = GENDERS[gender_net.forward()[0].argmax()]

            # Age
            age_net.setInput(blob)
            age_bucket = AGE_BUCKETS[age_net.forward()[0].argmax()]
            age = get_age_value(age_bucket)

            # Store detection (NO CSV YET)
            st.session_state.detections.append((age, gender))

            # Display logic
            if age >= 40:
                label = f"{gender}, Age: 60-100, Senior Citizen"
                color = (0, 255, 0)
            else:
                label = f"{gender}, Age: {age}"
                color = (255, 0, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        return frame


    # ================= UI =================
    option = st.radio("Select Input Source", ["Camera", "Upload Video"])
    frame_box = st.empty()

    # ================= CAMERA =================
    #if camera not working remove existing and add this same for sign detection
    #"""def open_camera():
    # Try local camera
    #for i in range(3):
      # cap = cv2.VideoCapture(i)
        #if cap.isOpened():
         #   st.success(f"✅ Local camera opened (index {i})")
          #  return cap

    # Fallback to IP camera
    #  IP_CAMERA_URL = "http://192.168.1.5:8080/video"
    #  cap = cv2.VideoCapture(IP_CAMERA_URL)

    # if cap.isOpened():
    #  st.success("✅ IP Camera connected")
    #   return cap

    #   return None

# cap = open_camera()

# if cap is None:
    # st.error("❌ No camera available (Local or IP)")
    # else:
    # while cap.isOpened():
    #  ret, frame = cap.read()
    #  if not ret:
    #     break
    # ================= CAMERA =================
    if option == "Camera":
        if st.button("▶ Start Camera"):
            cap = cv2.VideoCapture(0)    #

            if not cap.isOpened():
                st.error("❌ Camera not accessible")
            else:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = process_frame(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_box.image(frame, channels="RGB")

                cap.release()
                st.session_state.video_done = True

    # ================= VIDEO UPLOAD =================
    if option == "Upload Video":
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        st.info(" upload a video. Analysis Button will be available after the video finishes.")
        if video_file:
            st.session_state.detections = []
            st.session_state.video_done = False

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())

            cap = cv2.VideoCapture(tfile.name)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = process_frame(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_box.image(frame, channels="RGB")

            cap.release()
            st.session_state.video_done = True
            st.success("✅ Video processed. Click Analyze.")

    # ================= ANALYZE BUTTON =================
    if st.session_state.video_done:
        if st.button("🔍 Analyze Result"):
            seniors = [d for d in st.session_state.detections if d[0] >= 40]

            if seniors:
                age, gender = seniors[0]
                st.success("🟢 Senior Citizen Detected")

                df = pd.read_csv(CSV_FILE)
                df.loc[len(df)] = [
                    "60-100",
                    gender,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ]
                df.to_csv(CSV_FILE, index=False)
            else:
                st.info("🔵 No Senior Citizen Detected")

    # ================= CSV VIEW =================
    st.divider()
    st.subheader("📄 Senior Citizen Visit Log")
    st.dataframe(pd.read_csv(CSV_FILE), use_container_width=True)

# ================= TAB 4 ==========================
with tab4:
    # ================= LOAD MODELS =================
    gender_model = load_model("models/voice/gender_model.h5")
    age_model = load_model("models/voice/age_model.h5")
    emotion_model = load_model("models/voice/emotion_model.h5")

    # ---------------- TITLE ----------------
    st.markdown(
        "<p style='text-align:center;'>Voice-based Intelligent Analysis System</p>",
        unsafe_allow_html=True
    )

    # ---------------- TASK DESCRIPTION ----------------
    with st.expander("📌 Task Description (Click to Expand)"):
        st.write("""
        **Project Objective:**
        - Detect age from a voice note  
        - Accept only **male voices**  
        - Reject female voices  
        - If age > 60 → Detect emotion  
        - If age ≤ 60 → Only age detection  

        This project focuses on **logic-building and decision-based AI**.
        """)

    # ---------------- AUDIO UPLOAD ----------------
    audio = st.file_uploader("Upload voice (WAV)", type=["wav"])

    if audio:
        with open("temp.wav", "wb") as f:
            f.write(audio.read())

        # ---------------- FEATURE EXTRACTION ----------------
        features = extract_mfcc("temp.wav").reshape(1, -1)

        # ---------------- GENDER PREDICTION ----------------
        gender_pred = np.argmax(gender_model.predict(features))

        if gender_pred == 1:
            st.error("❌ Upload male voice.")
        else:
            # ---------------- AGE PREDICTION ----------------
            age = int(age_model.predict(features)[0][0])
            st.success(f"👨 Predicted Age: {age}")

            if age > 60:
                st.info("🧓 Senior Citizen Detected")

                # ---------------- EMOTION PREDICTION ----------------
                emotion_labels = ["Happy", "Sad", "Angry", "Neutral"]
                emotion = emotion_labels[
                    np.argmax(emotion_model.predict(features))
                ]
                st.success(f"😊 Emotion: {emotion}")

                # ================= PATH SETUP =================
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                MODEL_PATH = os.path.join(
                    BASE_DIR,
                    "models",
                    "sign_language",
                    "models",
                    "sign_language",
                    "sign_model.h5"
                )
                LABEL_PATH = os.path.join(
                    BASE_DIR,
                    "models",
                    "sign_language",
                    "labels.txt"
                )


# ================= TAB 5 : SIGN LANGUAGE ==================
with tab5:



    # ================= PATH SETUP =================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.join(
        BASE_DIR, "models", "sign_language", "sign_model.h5"
    )
    LABEL_PATH = os.path.join(
        BASE_DIR, "models", "sign_language", "labels.txt"
    )

    # ================= TIME RESTRICTION =================
    def is_allowed_time():
        hour = datetime.now().hour
        return 18 <= hour <= 22  # 6 PM – 10 PM

    # ================= PREPROCESS =================
    def preprocess_image(img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        img = img.reshape(1, 64, 64, 1) / 255.0
        return img

    # ================= LOAD MODEL SAFELY =================
    @st.cache_resource
    def load_sign_model():
        model = load_model(MODEL_PATH)
        with open(LABEL_PATH, "r") as f:
            labels = f.read().splitlines()
        return model, labels

    # ================= UI DESCRIPTION =================
    st.markdown("""
    This system recognizes **American Sign Language alphabets (A–Z)**  
    using a **CNN-based deep learning model**.
    """)

    # ================= TIME GATE (SAFE) =================
    if not is_allowed_time():
        st.warning("⏰ This feature is available only between **6 PM and 10 PM**.")
    else:
        # ================= FILE CHECK (SAFE) =================
        if not os.path.exists(MODEL_PATH):
            st.error("❌ sign_model.h5 not found")
        elif not os.path.exists(LABEL_PATH):
            st.error("❌ labels.txt not found")
        else:
            model, labels = load_sign_model()

            # ================= IMAGE UPLOAD =================
            st.subheader("📷 Upload Alphabet Sign Image")

            uploaded_file = st.file_uploader(
                "Upload hand sign image (A–Z)",
                type=["jpg", "jpeg", "png"]
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                img_array = np.array(image)

                st.image(image, caption="Uploaded Image", use_container_width=True)

                processed = preprocess_image(img_array)
                prediction = model.predict(processed)

                predicted_letter = labels[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                st.success(f"🧠 Predicted Alphabet: **{predicted_letter}**")
                st.info(f"📊 Confidence: **{confidence:.2f}%**")

            # ================= REAL-TIME CAMERA =================
            st.subheader("🎥 Real-Time Alphabet Detection")

            start_camera = st.checkbox("Start Camera")
            frame_window = st.image([])

            if start_camera:
                cap = cv2.VideoCapture(0)

                if not cap.isOpened():
                    st.error("❌ Camera not accessible")
                else:
                    ret, frame = cap.read()
                    if ret:
                        processed = preprocess_image(frame)
                        prediction = model.predict(processed)
                        predicted_letter = labels[np.argmax(prediction)]

                        cv2.putText(
                            frame,
                            f"Letter: {predicted_letter}",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (0, 255, 0),
                            3
                        )

                        frame_window.image(frame, channels="BGR")

                    cap.release()

# ================= TAB 6 ==========================

with tab6:
    # ---------------- LOAD YOLO MODEL ----------------
    # IMPORTANT: Make sure yolov8n.pt is in /models folder
    MODEL_PATH = "models/car/yolov8n.pt"

    if not os.path.exists(MODEL_PATH):
        st.error("❌ yolov8n.pt not found in models folder")
        st.stop()

    model = YOLO(MODEL_PATH)


    # ---------------- COLOR DETECTION FUNCTION ----------------
    def is_blue_car(car_img):
        """
        Detect if car is blue using HSV color space
        """
        hsv = cv2.cvtColor(car_img, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        blue_pixels = cv2.countNonZero(mask)
        total_pixels = car_img.shape[0] * car_img.shape[1]

        if total_pixels == 0:
            return False

        blue_ratio = blue_pixels / total_pixels
        return blue_ratio > 0.15  # threshold


    # ---------------- INPUT SELECTION ----------------
    input_type = st.radio("Select Input Type", ["Image", "Video"])

    # =========================================================
    # ================= IMAGE PROCESSING ======================
    # =========================================================
    if input_type == "Image":

        uploaded_image = st.file_uploader(
            "Upload Traffic Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image:
            image = Image.open(uploaded_image)
            frame = np.array(image)

            results = model(frame)

            car_count = 0
            people_count = 0

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if label == "car":
                        car_count += 1
                        car_crop = frame[y1:y2, x1:x2]

                        if car_crop.size == 0:
                            continue

                        # Color logic
                        if is_blue_car(car_crop):
                            box_color = (0, 0, 255)  # RED for blue cars
                            text = "Blue Car"
                        else:
                            box_color = (255, 0, 0)  # BLUE for other cars
                            text = "Car"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame, text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                    elif label == "person":
                        people_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Person", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            st.image(frame, caption="Processed Image", channels="BGR")

            col1, col2 = st.columns(2)
            col1.success(f"🚗 Total Cars: {car_count}")
            col2.success(f"🧍 Total People: {people_count}")

    # =========================================================
    # ================= VIDEO PROCESSING ======================
    # =========================================================
    else:

        uploaded_video = st.file_uploader(
            "Upload Traffic Video",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_video:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_video.read())

            cap = cv2.VideoCapture(temp_file.name)
            video_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)

                car_count = 0
                people_count = 0

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = model.names[cls]

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if label == "car":
                            car_count += 1
                            car_crop = frame[y1:y2, x1:x2]

                            if car_crop.size == 0:
                                continue

                            if is_blue_car(car_crop):
                                box_color = (0, 0, 255)
                            else:
                                box_color = (255, 0, 0)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                        elif label == "person":
                            people_count += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display counts
                cv2.putText(frame, f"Cars: {car_count}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.putText(frame, f"People: {people_count}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                video_placeholder.image(frame, channels="BGR")

            cap.release()
            os.unlink(temp_file.name)



# ================== TAB 7 : NATIONALITY-AWARE FACE ANALYSIS ==================
with tab7:


    st.header("🌍 Nationality-Aware Face Analysis System")

    # ================== PATH SETUP ==================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    NATIONALITY_MODEL_PATH = os.path.join(BASE_DIR, "models","nationality", "nationality_model.h5")
    EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "models","nationality", "emotion_model.h5")
    AGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "nationality","age_model.h5")


    # ================== LOAD MODELS ==================
    @st.cache_resource
    def load_all_models():
        nationality_model = load_model(NATIONALITY_MODEL_PATH)
        emotion_model = load_model(EMOTION_MODEL_PATH)
        age_model = load_model(AGE_MODEL_PATH)
        return nationality_model, emotion_model, age_model


    nationality_model, emotion_model, age_model = load_all_models()

    # ================== LABELS ==================
    NATIONALITIES = ["Indian", "United States", "African", "Other"]
    EMOTIONS = ["Angry", "Happy", "Sad", "Neutral", "Surprise", "Fear", "Disgust"]

    # ================== FACE DETECTOR ==================
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


    # ================== PREPROCESS FUNCTIONS ==================
    def preprocess_224(img):
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        return np.expand_dims(img, axis=0)


    def preprocess_64(img):
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        return np.expand_dims(img, axis=0)


    # ================== DRESS COLOR ==================
    def detect_dress_color(img):
        h, w, _ = img.shape
        lower_body = img[int(h * 0.6):h, :]
        avg_color = np.mean(lower_body, axis=(0, 1))

        b, g, r = avg_color
        if r > g and r > b:
            return "Red"
        elif g > r and g > b:
            return "Green"
        elif b > r and b > g:
            return "Blue"
        else:
            return "Mixed"


    # ================== UI ==================
    st.write(
        """
        **Features**
        - Nationality Detection
        - Emotion Recognition
        - Conditional Predictions based on Nationality
        """
    )

    uploaded_file = st.file_uploader(
        "📤 Upload an Image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Input Image")
            st.image(image, use_container_width=True)

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.error("❌ No face detected in the image.")
        else:
            with col2:
                st.subheader("🧠 Prediction Results")

                # Use only first detected face
                (x, y, w, h) = faces[0]
                face = img_array[y:y + h, x:x + w]

                # ================== NATIONALITY ==================
                nat_input = preprocess_224(face)
                nat_pred = nationality_model.predict(nat_input)
                nationality = NATIONALITIES[np.argmax(nat_pred)]

                # ================== EMOTION ==================
                emo_input = preprocess_64(face)
                emo_pred = emotion_model.predict(emo_input)
                emotion = EMOTIONS[np.argmax(emo_pred)]

                st.success(f"🌍 Nationality: **{nationality}**")
                st.info(f"😊 Emotion: **{emotion}**")

                # ================== CONDITIONAL LOGIC ==================
                st.markdown("---")
                st.subheader("🎯 Additional Predictions")

                if nationality == "Indian":
                    age_input = preprocess_224(face)
                    age = int(age_model.predict(age_input)[0][0])
                    dress = detect_dress_color(img_array)

                    st.write(f"🧓 Age: **{age} years**")
                    st.write(f"👕 Dress Colour: **{dress}**")

                elif nationality == "United States":
                    age_input = preprocess_224(face)
                    age = int(age_model.predict(age_input)[0][0])

                    st.write(f"🧓 Age: **{age} years**")

                elif nationality == "African":
                    dress = detect_dress_color(img_array)
                    st.write(f"👕 Dress Colour: **{dress}**")

                else:
                    st.write("ℹ No additional attributes predicted for this nationality.")