# Object Tracking using Lightweight Neural Network
Object Tracking using Lightweight Neural Network

📌 Overview
This project implements a lightweight hybrid object tracking system that combines deep learning-based object detection with classical tracking techniques. The system integrates YOLO11n for object detection and a Kalman Filter for motion-based tracking to achieve a balance between accuracy, stability, and real-time performance.
The project also includes a benchmark evaluation pipeline comparing the hybrid approach with traditional trackers such as KCF and CSRT using the OTB (Object Tracking Benchmark) dataset.

Features
🔍 Lightweight object detection using YOLO11n
🎯 Motion-based tracking using Kalman Filter
⚡ Real-time performance optimized for low-resource systems

Evaluation using standard OTB metrics:
Success (AUC)
Precision (@20px)
FPS (Frames Per Second)

Comparison with baseline trackers:
KCF (fast but less accurate)
CSRT (accurate but slower)

Methodology
The system follows a detect-then-track pipeline:

Detection: YOLO11n detects objects in the initial frame (and periodically)
Initialization: Bounding box initializes tracking
Tracking: Kalman Filter predicts object motion, KCF and CSRT used as baselines
Update: Kalman Filter corrects predictions using detections
Evaluation: Compared with ground truth using OTB metrics

Requirements
Install dependencies:
pip install opencv-python numpy matplotlib
For YOLO (Ultralytics):
pip install ultralytics
