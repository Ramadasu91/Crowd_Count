import streamlit as st
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv5 model for person detection
model = YOLO('yolov5s.pt')  # Pre-trained YOLO model for person detection

# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect people using YOLOv5 and faces using OpenCV
def detect_people_and_faces(frame):
    people_count = 0
    face_count = 0
    
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection using Haar cascade
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around detected faces and count them
    for (x, y, w, h) in faces:
        face_count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box for faces
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Perform person detection using YOLOv5
    results = model(frame)
    
    if results[0].boxes:
        for result in results[0].boxes.xyxy:
            if len(result) > 4:
                x1, y1, x2, y2 = map(int, result[:4])  # Bounding box coordinates
                class_id = int(result[5].item())  # Class ID
                
                # Check if detected object is a person (class_id == 0 for person in YOLOv5)
                if class_id == 0:
                    people_count += 1
                    # Draw bounding box around person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for person
                    cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Return the count of people and faces in the current frame
    current_frame_people_count = people_count + face_count
    return current_frame_people_count, frame

# Initialize session state variables for controls
if "run" not in st.session_state:
    st.session_state.run = False
if "pause" not in st.session_state:
    st.session_state.pause = False

# Streamlit app for real-time webcam feed with face and person detection
st.title("Real-time People & Face Detection with YOLOv5 and Stream Controls")
st.write("This app uses your webcam to detect people and faces in real-time.")

# Buttons for controls
start_button = st.button("Start Webcam")
pause_button = st.button("Pause/Resume")
stop_button = st.button("Stop Webcam")

# Placeholder to display the video feed and detection count
stframe = st.empty()
current_people_count_display = st.empty()

# Logic to handle controls
if start_button:
    st.session_state.run = True
    st.session_state.pause = False

if stop_button:
    st.session_state.run = False

if pause_button:
    st.session_state.pause = not st.session_state.pause  # Toggle pause state

# Start video capture when the user presses the "Start Webcam" button
if st.session_state.run:
    video_capture = cv2.VideoCapture(0)  # 0 is the default webcam

    if not video_capture.isOpened():
        st.error("Unable to access the camera")
    else:
        st.write("Webcam is running. Use the controls to pause, resume, or stop.")

        while st.session_state.run:
            ret, frame = video_capture.read()
            if not ret:
                break

            # If paused, do not process new frames, just display the last frame
            if not st.session_state.pause:
                # Detect people and faces in the current frame
                current_people_count, frame_with_detections = detect_people_and_faces(frame)

                # Convert the frame to RGB (Streamlit expects RGB format)
                frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)

                # Update the Streamlit frame and people count
                stframe.image(frame_pil, caption="Webcam Feed with Face and Person Detection", use_column_width=True)
                current_people_count_display.text(f"People detected in current frame: {current_people_count}")
            else:
                # If paused, continue showing the last frame without updating
                stframe.image(frame_pil, caption="Paused Frame", use_column_width=True)

        # Release the video capture object once done or stopped
        video_capture.release()
