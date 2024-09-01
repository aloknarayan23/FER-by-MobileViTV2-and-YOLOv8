## Step 1: Installing necessary libraries
!pip install ultralytics
!pip install torch torchvision
!pip install torchinfo


## Step 2: Importing librabries
import numpy as np
from torchinfo import summary
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ultralytics import YOLO
import time
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report



## Step 3: Loading the dataset

# Downloading FER-2013 dataset from Kaggle
!kaggle datasets download -d msambare/fer2013

# Unzipping datset to /content/FER_dataset_kaggle_format location
!unzip fer2013.zip -d /content/FER_dataset_kaggle_format



## Step 4: Data Preprocessing

# Data transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# Loading the dataset
train_dataset = datasets.ImageFolder(root='/content/train', transform=transform)
test_dataset = datasets.ImageFolder(root='/content/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)



## Step 5: Defining the Model and Training Loop

# Loading the yolo model for image classification
yolo_model = YOLO("yolov8s-cls.pt")

# Custom model class to modify the YOLOv8s model
class CustomYOLOv8s(nn.Module):
    def __init__(self, yolov8_model, num_classes=7):
        super(CustomYOLOv8s, self).__init__()
        self.backbone = yolov8_model.model.model[0:-1]  # Using all layers except the classifier

        self.custom_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.7),  # Dropout layer with a probability of 0.5
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.7),  # Dropout layer with a probability of 0.5
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)  # Passing through the backbone layers
        x = self.custom_layer(x)  # Passing through the custom layer
        x = self.classifier(x)  # Passing through the classifier
        return x

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initializing the custom model
custom_model = CustomYOLOv8s(yolo_model, num_classes=7).to(device)

custom_model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(custom_model.parameters(), lr=0.00001,  weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

summary(custom_model)

# Training loop with training accuracy
num_epochs = 50

for epoch in range(num_epochs):
    custom_model.train()  # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = custom_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct_train / total_train

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

    # Validating the model
    custom_model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    correct_test = 0
    total_test = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = custom_model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss = running_val_loss / len(test_loader.dataset)
    test_accuracy = correct_test / total_test
    print(f'Validation Loss: {val_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    scheduler.step()

# Saving the model state dictionary
torch.save(custom_model.state_dict(), 'custom_yolov8s_cls_Exp6.pt')



## Step 6: Evaluating the Model 

# Loading the model and set it to evaluation mode
model.load_state_dict(torch.load('custom_yolov8s_cls_Exp6.pt'))
model.to(device)
model.eval()

# Get the true labels and predictions
all_labels = []
all_predictions = []

# Predicting on test dataset
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Computing the confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Saving the confusion matrix as a png file
plt.savefig('YOLOv8s_cls_Exp6_confusion_matrix.png')  


# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Class names from the dataset
class_names = train_dataset.classes

# Generating a classification report
report = classification_report(all_labels, all_predictions, target_names=class_names)
print('\nClassification Report:')
print(report)

# Saving the classification report to a file
with open('YOLOv8s_cls_Exp6_classification_report.txt', 'w') as f:
    f.write(report)


## Step 7: Inference

# Testing with a sample image
image_path = "/content/image.jpg"

# Loading the image
image = Image.open(image_path)

# preprocessing the image and moving to device
image_tensor = transform(image).unsqueeze(0).to(device)  

# Get the timestamp before inference in seconds
start_ts = time.time()

# Predicting with the model
with torch.no_grad():  # Disable gradient calculation for inference
    emotion = model(image_tensor)

# Get the timestamp after the inference in seconds
end_ts = time.time()

# Printing the prediction
print(f"Predicted Emotion: {emotion}")

# Printing the time difference between start and end timestamps during prediction in seconds
print(f"Prediction Time [s]: {(end_ts - start_ts):.3f}")