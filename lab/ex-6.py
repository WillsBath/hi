# ==============================================================
# 1. MOUNT GOOGLE DRIVE
# ==============================================================
from google.colab import drive
drive.mount('/content/drive')

# ==============================================================
# 2. EXTRACT OR COPY DATASET FROM DRIVE
# ==============================================================

# If your dataset is a ZIP file (example: Alzheimer.zip)
# Uncomment this line:
# !unzip /content/drive/MyDrive/Alzheimer.zip -d /content/

# If your dataset is ALREADY a folder in Drive:
# !cp -r /content/drive/MyDrive/Alzheimer /content/

# After this, your test directory becomes:
# /content/Alzheimer/test



# ==============================================================
# 3. IMPORTS
# !pip install torch torchvision scikit-learn matplotlib numpy pillow
# ==============================================================
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np



# ==============================================================
# 4. CONFIG
# ==============================================================
test_dir = '/content/Alzheimer/test'   # use the copied/extracted dataset
img_size = 299
batch_size = 32
model_path = "/content/drive/MyDrive/Ex-6.h5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# ==============================================================
# 5. TRANSFORMS & LOAD TEST DATASET
# ==============================================================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = test_dataset.classes
print("Test Classes:", class_names)



# ==============================================================
# 6. LOAD TRAINED MODEL
# ==============================================================
from torchvision.models import Inception_V3_Weights
model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)

num_classes = len(class_names)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print(f" Model loaded successfully from: {model_path}")



# ==============================================================
# 7. EVALUATE MODEL
# ==============================================================
all_preds, all_labels = [], []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Inception returns (logits, aux); use logits
        if isinstance(outputs, tuple):
            outputs = outputs.logits  

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)



# ==============================================================
# 8. TEST ACCURACY
# ==============================================================
test_accuracy = 100 * correct / total
print(f"\n Test Accuracy: {test_accuracy:.2f}%")


# ==============================================================
# 9. CONFUSION MATRIX
# ==============================================================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - Alzheimer Test Dataset")
plt.show()



# ==============================================================
# 10. CLASSIFICATION REPORT
# ==============================================================
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))


#---------------------------------------------------------------------------------

#pip install torch torchvision pillow scikit-learn matplotlib numpy


from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import Inception_V3_Weights
import numpy as np

# ==============================================================
# 1. CONFIG
# ==============================================================
img_path = "/content/drive/MyDrive/test_img.jpg"   # <-- CHANGE THIS
model_path = "/content/drive/MyDrive/Ex-6.h5"      # your trained model
img_size = 299

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

class_names = ['Class1', 'Class2', 'Class3', 'Class4']  # <-- REPLACE with actual class names



# ==============================================================
# 2. LOAD MODEL
# ==============================================================
model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)

num_classes = len(class_names)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print("Model loaded successfully.")



# ==============================================================
# 3. TRANSFORMS
# ==============================================================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])



# ==============================================================
# 4. PREDICT SINGLE IMAGE
# ==============================================================
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

        if isinstance(outputs, tuple):
            outputs = outputs.logits

        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_class = class_names[pred.item()]
    confidence = conf.item() * 100

    print(f"\nPredicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    return predicted_class, confidence



# ==============================================================
# 5. RUN PREDICTION
# ==============================================================
predict_image(img_path)
