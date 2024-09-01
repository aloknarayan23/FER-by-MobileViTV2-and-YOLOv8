## Step 1: Installing necessary libraries
!pip install ultralytics



## Step 2: Importing librabries
from PIL import Image
import time
import numpy as np
from ultralytics import YOLO
from torchinfo import summary



## Step 3: Loading the dataset

# Downloading FER-2013 dataset from Kaggle
!kaggle datasets download -d msambare/fer2013

# Unzipping datset to /content/FER_dataset_kaggle_format location
!unzip fer2013.zip -d /content/FER_dataset_kaggle_format



## Step 4: Defining the Model and Training Loop

# Loading the trained yolo model from exp4 for image classification
model = YOLO("yolov8s_cls_Exp5_best.pt")

summary(model)

# Train the model
results = model.train(data="/content/FER_dataset_kaggle_format", epochs=50, imgsz=48, batch = 64, dropout=0.2, freeze= 5, device=0)
# Results including best model, classification report and confusion matrix saved to runs/classify/train



## Step 5: Inference

# defining path of a sample image
image_path = "/content/image.jpg"

# Get the timestamp before inference in seconds
start_ts = time.time()

# Predicting on a sample image
emotion = model(image_path)

# Get the timestamp after the inference in seconds
end_ts = time.time()

# Printing the prediction
print(f"Predicted Emotion: {emotion}")

# Printing the time difference between start and end timestamps during prediction in seconds
print(f"Prediction Time [s]: {(end_ts - start_ts):.3f}")
