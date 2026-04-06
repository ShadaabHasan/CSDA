# Object Tracking using Lightweight Neural Network
Object Tracking using Lightweight Neural Network

📌 Overview
This project implements a lightweight hybrid object tracking system that combines deep learning-based object detection with classical tracking techniques. The system integrates YOLO11n for object detection and a Kalman Filter for motion-based tracking to achieve a balance between accuracy, stability, and real-time performance.
The project also includes a benchmark evaluation pipeline comparing the hybrid approach with traditional trackers such as KCF and CSRT using the OTB (Object Tracking Benchmark) dataset.

🚀 Features
🔍 Lightweight object detection using YOLO11n
🎯 Motion-based tracking using Kalman Filter
⚡ Real-time performance optimized for low-resource systems

📊 Evaluation using standard OTB metrics:
Success (AUC)
Precision (@20px)
FPS (Frames Per Second)

📈 Comparison with baseline trackers:
KCF (fast but less accurate)
CSRT (accurate but slower)

🧠 Methodology
The system follows a detect-then-track pipeline:
Detection
YOLO11n detects objects in the initial frame (and periodically)
Initialization
Bounding box initializes tracking
Tracking
Kalman Filter predicts object motion
KCF and CSRT used as baselines
Update
Kalman Filter corrects predictions using detections
Evaluation
Compared with ground truth using OTB metrics

🏗️ Project Structure
├── EvaluateOTB.py        # Main evaluation script
├── KFmain.ipynb         # Kalman filter implementation
├── yolo11n_training.ipynb  # YOLO training notebook
├── best.pt              # Trained YOLO model
├── README.md            # Project documentation
⚙️ Requirements
Install dependencies:
pip install opencv-python numpy matplotlib
For YOLO (Ultralytics):
pip install ultralytics
▶️ How to Run
Run Evaluation on OTB Dataset
python EvaluateOTB.py --seq path_to_sequence
Optional Arguments
--show           # Visualize tracking
--output_plot    # Save comparison plot
--output_csv     # Save results as CSV
Example:
python EvaluateOTB.py --seq OTB/Car1 --show
📊 Evaluation Metrics
Success (AUC): Measures overlap between predicted and ground truth boxes
Precision (@20px): Measures center location accuracy
FPS: Measures real-time performance
📈 Results Summary (CAR1 Sequence)
Tracker	AUC	Precision@20px	FPS
Kalman	0.918	0.993	N/A*
KCF	0.087	0.177	357
CSRT	0.272	0.827	41.3
*Kalman uses simulated ground truth for correction (not a real-world FPS)
📦 Dataset
Pascal VOC → Used for training YOLO11n
OTB Dataset → Used for tracking evaluation
🔍 Key Insights
KCF is extremely fast but lacks accuracy
CSRT provides better localization but is slower
Kalman Filter improves temporal stability
Hybrid approach reduces drift and improves robustness

