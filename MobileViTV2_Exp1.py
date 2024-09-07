## Step 1: Installing necessary libraries
!pip install --upgrade transformers
!pip install torch torchvision scikit-learn
!pip install --upgrade torch torchvision torchaudio
!pip install datasets 
!pip install accelerate -U --quiet



## Step 2: Importing libraries
import accelerate
import os
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image
import numpy as np
import torch
from torchvision import transforms
from datasets import load_from_disk
from transformers import AutoImageProcessor, MobileViTV2ForImageClassification, TrainingArguments, Trainer
import pickle
import joblib
from PIL import Image
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns



## Step 3: Loading the dataset

# Downloading FER-2013 dataset from Kaggle
!kaggle datasets download -d msambare/fer2013

# Unzipping datset to /content/FER_dataset_kaggle_format location
!unzip fer2013.zip -d /content/FER_dataset_kaggle_format

### Step 3.1: Converting FER-2013 dataset from Kaggle into hugging face dataset format.

# Defining the path to the dataset
dataset_path = 'FER_dataset_kaggle_format'

# Defining the class labels
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']   
class_label = ClassLabel(names=class_names)

# Function to load images and labels
def load_images_labels(folder):  
    data = {'image': [], 'label': []}
    for label in class_names:
        folder_path = os.path.join(dataset_path, folder, label)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg'):
                file_path = os.path.join(folder_path, file_name)
                data['image'].append(file_path)
                data['label'].append(class_label.str2int(label))
    return data

# Loading train and validation
train_data = load_images_labels('train')    
val_data = load_images_labels('test')

# Creating Dataset objects
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

# Defining features
features = Features({
                    'image': Image(decode=True),
                    'label': class_label
                    })

# Set features
train_dataset = train_dataset.cast(features)
val_dataset = val_dataset.cast(features)

# Creating a DatasetDict
dataset_dict = DatasetDict({
                            'train': train_dataset,
                            'validation': val_dataset,
                            })

# Saving dataset locally
dataset_dict.save_to_disk('fer2013_hf_new')     

# Loading the dataset
dataset = load_from_disk('fer2013_hf_new')



## Step 4: Data Preprocessing

from torch.utils.data import DataLoader, Dataset

# Initializing the image processor
image_processor = AutoImageProcessor.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")

# Function to apply a Gaussian filter to an RGB image
def apply_gaussian_filter(img, kernel_size=5, sigma=1.0):
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    x_grid = x.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    gaussian_kernel = (1 / (2.0 * torch.pi * sigma ** 2)) * torch.exp(
        -torch.sum(xy_grid ** 2, dim=-1) / (2 * sigma ** 2)
    )
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # Making the kernel compatible with RGB images (3 channels)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)  # Repeats for 3 RGB channels

    # Apply the Gaussian filter to each RGB channel independently
    img = img.unsqueeze(0)  # Adding batch dimension
    filtered_img = F.conv2d(img, gaussian_kernel, padding=kernel_size // 2, groups=3)
    return filtered_img.squeeze(0)  # Removes batch dimension

# Defining augmentation transformations
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally with a probability of 0.5
    transforms.RandomRotation(degrees=90),   # Randomly rotate images by 90 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random shifting
    transforms.ToTensor(),  # Convert image to tensor (required for filtering noise)
    transforms.Lambda(lambda img: apply_gaussian_filter(img)),  # Add Gaussian filter 
    transforms.ToPILImage(),  # Convert back to PIL Image
])

# Defining a custom dataset class to preprocess the images
class FER2013Dataset(Dataset):
    def __init__(self, split, image_processor, augmentations=None):
        self.dataset = dataset[split]
        self.image_processor = image_processor
        self.augmentations = augmentations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        
        if self.augmentations:
            image = self.augmentations(image)        


        processed_image = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        return {'pixel_values': processed_image, 'labels': label}

# Creating DataLoader for train, validation, and test sets
train_dataset = FER2013Dataset('train', image_processor, augmentations=augmentation_transforms)
val_dataset = FER2013Dataset('validation', image_processor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



## Step 5: Defining the Model and Training Loop

# Loading the model for image classification from hugging face
model = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256", num_labels=7,ignore_mismatched_sizes=True)

# Defining the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=1e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    num_train_epochs=50,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Defining the compute_metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Initializing the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Training the model
trainer.train()

# saving model locally
joblib.dump(model, 'MViTV2_Exp1.pkl') 



## Step 6: Evaluating the Model 

# Evaluating on the validation set
val_results = trainer.evaluate(eval_dataset=val_dataset)
print(f"Validation Accuracy: {val_results['eval_accuracy']:.2f}")

# Moving the model to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

y_true = []
y_pred = []

# Iterating over the validation dataset
for sample in val_dataset:
    inputs = {'pixel_values': sample['pixel_values'].unsqueeze(0).to(device)}
    
    # Performing prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    y_pred.append(predicted_class_idx)
    y_true.append(sample['labels'])

# Number of classes in dataset
num_labels = 7  

# Creating mappings from class names to indices and vice versa
label_to_index = {label: idx for idx, label in enumerate(class_names)}
index_to_label = {idx: label for idx, label in enumerate(class_names)}

# Generating the confusion matrix
cm = confusion_matrix(y_pred,y_true, labels=range(num_labels))

# Plotting the confusion matrix using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Confusion Matrix')

# Saving the confusion matrix as a png file
plt.savefig('MViTV2_Exp1_confusion_matrix.png')  

# Generating the classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Saving the classification report to a file
with open('MViTV2_Exp1_classification_report.txt', 'w') as f:
    f.write(report)



## Step 7: Inference

# Function to predict emotion from an image
def predict_emotion(image_path, model, image_processor):
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(image, return_tensors="pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return dataset['train'].features['label'].int2str(predicted_class_idx)

# defining path of a sample image
image_path = "/content/image.jpg"

# Get the timestamp before inference in seconds
start_ts = time.time()

# Predicting on a sample image
emotion = predict_emotion(image_path, model, image_processor)

# Get the timestamp after the inference in seconds
end_ts = time.time()

# Printing the prediction
print(f"Predicted Emotion: {emotion}")

# Printing the time difference between start and end timestamps during prediction in seconds
print(f"Prediction Time [s]: {(end_ts - start_ts):.3f}")



