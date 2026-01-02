import cv2

net = cv2.dnn.readNetFromCaffe(
    r"C:\Users\A\OneDrive\Desktop\jupyter project\multi_modal_ai_project\models\face\deploy.prototxt",
    r"C:\Users\A\OneDrive\Desktop\jupyter project\multi_modal_ai_project\models\face\res10_300x300_ssd_iter_140000.caffemodel"
)

print("Face model loaded OK")
