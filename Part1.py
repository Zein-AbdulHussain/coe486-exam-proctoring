import kagglehub

# Download latest version
path = kagglehub.dataset_download("maulidio16/300w-lp")

print("Path to dataset files:", path)

import os

#find location
for root, dirs, files in os.walk(path):
    print("ROOT:", root)
    print("DIRS:", dirs[:5])
    print("FILES:", files[:5])
    print("-"*40)
    break

import glob

images = glob.glob(path + "/**/*.jpg", recursive=True)

print("Total images found:", len(images))

if len(images) > 0:
    print("Sample image:", images[0])

path = kagglehub.dataset_download("maulidio16/300w-lp")
print(path)

import scipy.io as sio
import cv2
import numpy as np

def load_sample(img_path):
    mat_path = img_path.replace(".jpg", ".mat")

    mat = sio.loadmat(mat_path)

    pose = mat["Pose_Para"][0]

    pitch, yaw, roll = np.degrees(pose[:3])

    img = cv2.imread(img_path)

    return img, (pitch, yaw, roll)

img_path = None

for root, _, files in os.walk(path):
    for f in files:
        if f.endswith(".jpg"):
            img_path = os.path.join(root, f)
            break
    if img_path:
        break

print("Using image:", img_path)

import scipy.io as sio
import cv2
import numpy as np

def load_sample(img_path):
    mat_path = img_path.replace(".jpg", ".mat")

    mat = sio.loadmat(mat_path)
    pose = mat["Pose_Para"][0]

    pitch, yaw, roll = np.degrees(pose[:3])

    img = cv2.imread(img_path)

    return img, (pitch, yaw, roll)

img, angles = load_sample(images[0])

print("Pitch, Yaw, Roll:", angles)

from google.colab.patches import cv2_imshow
cv2_imshow(img)

import numpy as np
import scipy.io as sio
import cv2

X_images = []
y_angles = []

for img_path in images[:2000]:  # start small first
    mat_path = img_path.replace(".jpg", ".mat")

    try:
        mat = sio.loadmat(mat_path)
        pose = mat["Pose_Para"][0]

        pitch, yaw, roll = np.degrees(pose[:3])

        img = cv2.imread(img_path)

        if img is not None:
            X_images.append(img_path)
            y_angles.append([pitch, yaw, roll])

    except:
        continue

print("Loaded samples:", len(X_images))

import numpy as np
import cv2
import scipy.io as sio

X = []
y = []

for img_path in images[:2000]:  # IMPORTANT: start small
    try:
        mat_path = img_path.replace(".jpg", ".mat")
        mat = sio.loadmat(mat_path)

        pose = mat["Pose_Para"][0]
        pitch, yaw, roll = np.degrees(pose[:3])

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (224, 224))

        X.append(img)
        y.append([pitch, yaw, roll])

    except:
        continue

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X = X / 255.0  # normalize images

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

