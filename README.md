# COE486 Intelligent Exam Proctoring

Computer vision final project for COE486, Spring 2026. The system detects suspicious online-exam behavior using webcam-based face landmarks, gaze/head-pose estimation, YOLOv8 phone detection, and a sliding-window suspicion score.

## Repository Contents

This repository stores the submitted code as two Git LFS zip archives:

- `Vision.zip` - integrated live proctoring demo with MediaPipe landmarks, iris-ratio gaze, CNN head pose, YOLOv8 phone detection, and the 90-frame suspicion aggregator.
- `person_b_phone_detection.zip` - phone-detection and head-pose training/testing workspace, including Roboflow datasets, saved YOLO runs, trained weights, and evaluation outputs.

Because the archives are large, install Git LFS before cloning or pulling the repository.

## 1. Clone the Repository

```bash
git lfs install
git clone https://github.com/Zein-AbdulHussain/coe486-exam-proctoring.git
cd coe486-exam-proctoring
git lfs pull
```

If the zip files look very small after cloning, Git LFS did not download the real files. Run:

```bash
git lfs pull
```

## 2. Extract the Code

### macOS / Linux

```bash
unzip Vision.zip
unzip person_b_phone_detection.zip
```

### Windows PowerShell

```powershell
Expand-Archive -Path Vision.zip -DestinationPath .
Expand-Archive -Path person_b_phone_detection.zip -DestinationPath .
```

After extraction, the expected folders are:

```text
Vision/
person_b_phone_detection/
```

If the second archive extracts into a folder named `person_b_phone_detection 5`, open that folder and use the inner `person_b_phone_detection` directory.

## 3. Run the Integrated Proctoring Demo

This is the main final-project demo.

### macOS / Linux

```bash
cd Vision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python proctor_full.py
```

### Windows PowerShell

```powershell
cd Vision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python proctor_full.py
```

Demo controls:

- Press `c` while looking straight at the screen to calibrate gaze.
- Press `q` to quit.

The integrated demo loads:

- `face_landmarker.task` for MediaPipe face and iris landmarks.
- `head_pose_5000.keras` for CNN pitch/yaw/roll prediction.
- `yolov8n.pt` for YOLOv8 COCO phone detection.
- A 90-frame sliding window to convert frame-level suspicious flags into a stable suspicion score.

## 4. Run Individual Vision Modules

From inside the `Vision/` folder with the virtual environment active:

```bash
python head_pose.py
python head_pose_cnn.py
```

`head_pose.py` runs the solvePnP geometric head-pose baseline. `head_pose_cnn.py` runs the trained CNN head-pose module.

## 5. Set Up Phone Detection and Training Workspace

Go to the phone-detection workspace:

### macOS / Linux

```bash
cd ../person_b_phone_detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow scipy scikit-learn matplotlib kagglehub
```

### Windows PowerShell

```powershell
cd ..\person_b_phone_detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow scipy scikit-learn matplotlib kagglehub
```

Verify the installation:

```bash
python -m src.check_install
```

## 6. Roboflow Dataset Setup

The phone-detection code uses Roboflow datasets. If the datasets are already included after unzipping, this step can be skipped. To download them again, create a `.env` file:

### macOS / Linux

```bash
cp .env.example .env
```

### Windows PowerShell

```powershell
copy .env.example .env
```

Edit `.env` and add a valid Roboflow API key:

```text
ROBOFLOW_API_KEY=your_key_here
CLASSROOM_WORKSPACE=computervisionprojects-siakl
CLASSROOM_PROJECT=classroom-cell-phone-detection
CLASSROOM_VERSION=5
MOBILE_WORKSPACE=exam-detection-a9bsf
MOBILE_PROJECT=mobile-phone-detection-mtsje
MOBILE_VERSION=1
```

Then download both datasets:

```bash
python -m src.download_roboflow_datasets --only both --format yolov8 --output-dir data/roboflow
```

Do not commit real API keys to GitHub. Use `.env.example` for placeholders and `.env` only for local execution.

## 7. Test the COCO Phone Baseline

Run YOLOv8 COCO phone detection on a live webcam:

```bash
python -m src.webcam_coco_baseline --source 0 --model yolov8n.pt --conf 0.25
```

If the webcam does not open, try another camera index:

```bash
python -m src.webcam_coco_baseline --source 1 --model yolov8n.pt --conf 0.25
```

To save a demo video:

```bash
python -m src.webcam_coco_baseline --source 0 --model yolov8n.pt --conf 0.25 --save-video runs/webcam_baseline.mp4
```

## 8. Evaluate the COCO Baseline on Dataset Images

Classroom validation set:

```bash
python -m src.evaluate_coco_baseline_on_yolo_dataset \
  --dataset data/roboflow/classroom-cell-phone-detection \
  --split valid \
  --model yolov8n.pt \
  --conf 0.25 \
  --match-iou 0.50 \
  --output-dir runs/coco_baseline_eval
```

Mobile-phone validation set:

```bash
python -m src.evaluate_coco_baseline_on_yolo_dataset \
  --dataset data/roboflow/mobile-phone-detection-1674 \
  --split valid \
  --model yolov8n.pt \
  --conf 0.25 \
  --match-iou 0.50 \
  --output-dir runs/coco_baseline_eval
```

Outputs are saved as CSV files under:

```text
runs/coco_baseline_eval/
```

## 9. Train YOLOv8 Phone Detectors

Train on the classroom phone dataset:

```bash
yolo detect train model=yolov8n.pt data=data/roboflow/classroom-cell-phone-detection/data.yaml epochs=50 batch=8 imgsz=640 device=cpu name=phone_classroom_yolov8n
```

Continue training on the mobile-phone dataset using the classroom-trained weights:

```bash
yolo detect train model=runs/detect/phone_classroom_yolov8n/weights/best.pt data=data/roboflow/mobile-phone-detection-1674/data.yaml epochs=30 batch=8 imgsz=640 device=cpu name=phone_mobile_yolov8n
```

If CUDA or Apple Silicon acceleration is available, replace `device=cpu` with the appropriate device setting. Training results are written to:

```text
runs/detect/<run_name>/
```

Important outputs:

```text
runs/detect/<run_name>/weights/best.pt
runs/detect/<run_name>/weights/last.pt
runs/detect/<run_name>/results.csv
runs/detect/<run_name>/results.png
```

## 10. Validate a Trained YOLO Model

Example validation command:

```bash
yolo detect val model=runs/detect/phone_classroom_yolov8n/weights/best.pt data=data/roboflow/classroom-cell-phone-detection/data.yaml imgsz=640 batch=8 device=cpu
```

For the final online-exam model, validate the final saved weights against the corresponding YOLO-format dataset:

```bash
yolo detect val model=runs/detect/phone_online_exam_yolov8n/weights/best.pt data=/path/to/online_exam_proctoring/data.yaml imgsz=640 batch=8 device=cpu
```

## 11. Train and Test the Head-Pose CNN

The head-pose script trains on the 300W-LP dataset using `kagglehub`, saves the best model, saves training history, plots training vs. validation MAE, and writes test predictions.

Run from inside `person_b_phone_detection/`:

```bash
python src/head_pose_local_5000.py
```

Outputs:

```text
runs/head_pose_5000/head_pose_5000.keras
runs/head_pose_5000/head_pose_5000_final.keras
runs/head_pose_5000/training_history.csv
runs/head_pose_5000/mae_curve.png
runs/head_pose_5000/test_predictions.csv
```

To test the trained head-pose model with a webcam:

```bash
python -m src.test_head_pose_webcam
```

## 12. Expected Demo Order

For grading or reproduction, use this order:

1. Clone with Git LFS and unzip both archives.
2. Run `Vision/proctor_full.py` to show the complete live proctoring system.
3. Run `src.webcam_coco_baseline` to show phone detection alone.
4. Run YOLO validation commands to reproduce phone-detection metrics.
5. Run `src/head_pose_local_5000.py` only if retraining the head-pose CNN is required.
6. Inspect saved metrics and plots in `runs/head_pose_5000/` and `runs/detect/`.

## Troubleshooting

If `python` is not found on macOS or Linux, use `python3`.

If `cv2.VideoCapture(0)` does not open the webcam, try source `1` or `2`.

If Roboflow download fails, check the API key and dataset version values in `.env`.

If TensorFlow installation fails, use Python 3.10 or 3.11. Some TensorFlow builds may not support the newest Python version on every platform.

If GitHub downloads pointer files instead of the real zip files, run `git lfs pull`.
