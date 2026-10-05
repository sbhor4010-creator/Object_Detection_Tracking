# CodeAlpha - Object Detection and Tracking

## Project Overview

This project implements a real-time object detection and tracking system using a webcam.

The system uses YOLO for object detection and Deep SORT for tracking detected objects across video frames.

This project was developed as part of the CodeAlpha Artificial Intelligence Internship.

## Task

**Task 4 - Object Detection and Tracking**

## Technologies Used

- Python
- OpenCV
- YOLO
- Deep SORT
- NumPy

## Features

- Real-time webcam video input
- Object detection using YOLO
- Bounding boxes around detected objects
- Object labels
- Confidence scores
- Object tracking using Deep SORT
- Unique tracking IDs
- Real-time object counting
- FPS display
- Processed video recording

## How It Works

1. The webcam captures real-time video frames using OpenCV.
2. YOLO processes each frame to detect objects.
3. Bounding boxes, confidence scores, and class IDs are extracted.
4. The detected objects are passed to the Deep SORT tracker.
5. Deep SORT assigns unique tracking IDs to objects.
6. Bounding boxes and tracking IDs are displayed on the video.
7. The system displays the number of detected objects and FPS.
8. The processed video is saved as `output.mp4`.

## Project Structure

```text
Object_Detection_Tracking/
│
├── main.py
├── README.md
├── requirements.txt
├── output.mp4
└── yolo11n.pt
## Installation

Install the required Python packages:

```bash
pip install ultralytics opencv-python numpy deep-sort-realtime

Then continue with:

```markdown
## Running the Project

Open the project folder in VS Code and run:

```bash
python main.py
The webcam will open automatically.

Press Q to stop the program.

So **yes, keep the Installation section** in your README.