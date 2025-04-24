import streamlit as st
import cv2
import numpy as np
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Function to download model from S3
def download_model_from_s3(url, filename):
    if not os.path.exists(filename):
        with st.spinner("Downloading model from S3..."):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            else:
                st.error("Failed to download model from S3.")
                st.stop()

# Use your actual S3 URL
S3_MODEL_URL = "https://parthanium.s3.ap-south-1.amazonaws.com/ferNetModel.h5"
MODEL_FILENAME = "ferNetModel.h5"

# Download model
download_model_from_s3(S3_MODEL_URL, MODEL_FILENAME)

# Load face detector and emotion classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model(MODEL_FILENAME)
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','sad','Surprise']

# Streamlit UI
st.title("Real-time Emotion Detector")

run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    success, frame = cap.read()
    if not success:
        st.error("Failed to access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()
cv2.destroyAllWindows()
