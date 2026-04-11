# COE-486 Intelligent Exam Proctoring

Computer Vision project for COE-486, American University of Sharjah, Spring 2026.
Instructor: Dr. Omar Arif.

## Team
- [Zein Abdul-Hussain]
- [Mansoor Kamkar]
- [Yazan Abu Jbarah]

## Overview
A lightweight exam proctoring system combining gaze/head pose attention
tracking with YOLOv8-based prohibited object detection. Runs locally on a
standard webcam.

## Modules
- `phone_detection/` — YOLOv8 fine-tuned for phone detection
- `gaze_tracking/` — MediaPipe + CNN-based gaze and head pose estimation
- `integration/` — sliding-window aggregator and live demo
- `data/custom/` — simulated exam footage recorded at AUS
- `report/` — final IEEE-format report

## Setup
\`\`\`bash
pip install -r phone_detection/requirements.txt
\`\`\`

## Phone Detection (Baseline)
\`\`\`bash
cd phone_detection
python baseline.py
\`\`\`

## Phone Detection (Fine-tune)
\`\`\`bash
python download_data.py        # add your Roboflow API key first
python train_and_eval.py
python live_demo.py
\`\`\`

## Datasets
- Classroom Cell Phone Detection (Roboflow)
- Mobile Phone Detection (Roboflow, 1,674 images)
- MPIIFaceGaze (gaze estimation)
- 300W-LP / AFLW2000-3D (head pose)
- Custom AUS exam simulation footage
