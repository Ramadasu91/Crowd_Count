# File: app.py

import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import tempfile
import os
import numpy as np
from io import BytesIO

# Load YOLOv5 model
model = YOLO('yolov5s.pt')  # Load YOLOv5s model

# Function to detect people and draw bounding boxes
def detect_and_count_people(frame):
    results = model(frame)  # Perform detection on the frame
    people_count = 0

    # Get bounding boxes, class IDs, and confidence scores
    boxes = results[0].boxes.xyxy  # Bounding box coordinates
    confs = results[0].boxes.conf  # Confidence scores
    classes = results[0].boxes.cls  # Class IDs

    # Iterate over detections
    for i in range(len(boxes)):
        class_id = int(classes[i].item())  # Get class ID
        if class_id == 0:  # Class ID 0 corresponds to 'person' in YOLOv5
            people_count += 1

            # Get bounding box coordinates and confidence score
            x1, y1, x2, y2 = map(int, boxes[i])  # Bounding box coordinates
            confidence = confs[i].item()  # Confidence score

            # Draw bounding box and confidence score on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for person

    return frame, people_count

# Streamlit application
def main():
    # Streamlit UI
    st.set_page_config(page_title="Skavch Crowd Count Engine", layout="wide")

    # Add an image to the header
    st.image("bg1.jpg", use_column_width=True)  # Adjust the image path as necessary
    st.title("Skavch Crowd Count Engine")

    # File uploader to upload a video
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        # Store the uploaded video in-memory using BytesIO
        video_bytes = BytesIO(video_file.read())

        st.text(f"Processing video...")

        # Open video file using OpenCV from in-memory bytes
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        with open(temp_video_path, 'wb') as f:
            f.write(video_bytes.read())
        
        video_capture = cv2.VideoCapture(temp_video_path)

        # Get video properties
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create an in-memory video file to store the output
        output_video = BytesIO()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec to 'mp4v' for better compatibility
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

        # Process each frame in the video
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Detect people and get frame with bounding boxes
            frame, people_count = detect_and_count_people(frame)

            # Display the people count on the top left corner of the frame
            cv2.putText(frame, f'People Count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write the processed frame to the output video
            video_writer.write(frame)

        # Release resources
        video_capture.release()
        video_writer.release()

        st.text(f"Video processing complete!")

        # Read the processed video and display it in Streamlit
        with open(temp_video_path, 'rb') as f:
            video_bytes = f.read()

        st.video(video_bytes)

        # Provide a download button for the output video
        st.download_button('Download Processed Video', data=video_bytes, file_name="output_video.mp4")

if __name__ == '__main__':
    main()
