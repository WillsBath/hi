# ==============================================================
# 1. MOUNT GOOGLE DRIVE
# ==============================================================
from google.colab import drive
drive.mount('/content/drive')

# ==============================================================
# 2. EXTRACT OR COPY DATASET FROM DRIVE
# ==============================================================

# If your dataset is a ZIP FILE (example: Alzheimer.zip)
# UNCOMMENT this line:
# !unzip /content/drive/MyDrive/Alzheimer.zip -d /content/

# If your dataset is ALREADY A FOLDER in Drive:
# !cp -r /content/drive/MyDrive/Alzheimer /content/

# After this, the dataset path becomes:
# /content/Alzheimer/test


# ==============================================================
# 3. IMPORT LIBRARIES
# pip install torch torchvision scikit-learn matplotlib numpy pillow
# ==============================================================
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np


# ==============================================================
# 4. CONFIG
# ==============================================================
test_dir = '/content/Alzheimer/test'   # <<-- USE THE TEST FOLDER
batch_size = 32
img_size = 299

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==============================================================
# 5. TRANSFORMS & LOAD ONLY TEST DATA
# ==============================================================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = test_dataset.classes
num_classes = len(class_names)
print("Detected Test Classes:", class_names)


# ==============================================================
# 6. LOAD InceptionV3 MODEL
# ==============================================================
from torchvision.models import Inception_V3_Weights

model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)

# Replace final fully-connected layers
model.fc = nn.Linear(model.fc.in_features, num_classes)

if model.AuxLogits is not None:
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)

model = model.to(device)


# ==============================================================
# 7. LOAD TRAINED WEIGHTS
# ==============================================================
model.load_state_dict(torch.load("/content/drive/MyDrive/Ex-5.h5", map_location=device))
print("✅ Model loaded successfully!")


# ==============================================================
# 8. EVALUATE ON TEST SET
# ==============================================================
model.eval()
all_preds = []
all_labels = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        # Inception returns (logits, aux); use logits
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)


# ==============================================================
# 9. TEST ACCURACY
# ==============================================================
accuracy = 100 * correct / total
print(f"\n✅ Test Accuracy: {accuracy:.2f}%")


# ==============================================================
# 10. CONFUSION MATRIX
# ==============================================================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - Alzheimer Test Dataset")
plt.show()


# ==============================================================
# 11. CLASSIFICATION REPORT
# ==============================================================
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))
