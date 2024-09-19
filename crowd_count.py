import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

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
    
    # Return the total count of people (both detected faces and persons)
    total_people_count = people_count + face_count
    return total_people_count, frame

# Initialize session state variables
if "run" not in st.session_state:
    st.session_state.run = False
if "pause" not in st.session_state:
    st.session_state.pause = False
if "frame" not in st.session_state:
    st.session_state.frame = None

# Streamlit app for webcam input with face and person detection
st.title("Webcam Face & Person Detection with YOLOv5")

# Buttons for controls
start_button = st.button("Start Webcam")
pause_button = st.button("Pause")
stop_button = st.button("Stop Webcam")

# Webcam input
camera_input = st.camera_input("Capture Image", key="camera_input")

# Logic to handle controls
if start_button:
    st.session_state.run = True
    st.session_state.pause = False

if stop_button:
    st.session_state.run = False
    st.session_state.pause = False
    st.session_state.frame = None  # Clear the frame when stopped

if pause_button:
    st.session_state.pause = not st.session_state.pause  # Toggle pause state

if camera_input:
    # Convert the captured image to OpenCV format
    image = Image.open(camera_input)
    frame = np.array(image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if st.session_state.run:
        # If not paused, process the frame
        if not st.session_state.pause:
            # Detect people and faces in the current frame
            people_count, frame_with_detections = detect_people_and_faces(frame_bgr)

            # Convert the frame to RGB for display in Streamlit
            frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            # Display the frame with detections
            st.image(frame_pil, caption="Detected People & Faces", use_column_width=True)
            st.write(f"People detected: {people_count}")

    st.session_state.frame = frame
