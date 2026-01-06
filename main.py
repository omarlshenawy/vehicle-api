from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import io

app = FastAPI()

# =====================
# Labels & Categories
# =====================
labels = {
    0: "Bicycle", 1: "Boat", 2: "Bus", 3: "Car", 4: "Helicopter",
    5: "Minibus", 6: "Motorcycle", 7: "Taxi", 8: "Train", 9: "Truck"
}

category_map = {
    "Bicycle": "Land vehicle",
    "Motorcycle": "Land vehicle",
    "Car": "Personal land vehicle",
    "Taxi": "Public transport vehicle",
    "Bus": "Public transport vehicle",
    "Minibus": "Public transport vehicle",
    "Truck": "Commercial land vehicle",
    "Boat": "Water vehicle",
    "Train": "Rail vehicle",
    "Helicopter": "Air vehicle"
}

# =====================
# Image Transform
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# Load Model (ON STARTUP)
# =====================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("vehicle_model.pth", map_location="cpu"))
model.eval()

# =====================
# Color Detection
# =====================
def hue_to_color(h, s, v):
    if v < 50: return "Black"
    elif s < 50 and v > 200: return "White"
    elif s < 50: return "Gray"
    elif h < 10 or h >= 170: return "Red"
    elif h < 22: return "Orange"
    elif h < 33: return "Yellow"
    elif h < 78: return "Green"
    elif h < 130: return "Blue"
    elif h < 160: return "Purple"
    else: return "Unknown"

def get_vehicle_color(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    pixels = img.reshape(-1, 3)
    pixels = [p for p in pixels if p[1] > 40]
    if not pixels:
        return "Unknown"
    avg = np.mean(pixels, axis=0).astype(int)
    return hue_to_color(avg[0], avg[1], avg[2])

# =====================
# API Endpoint
# =====================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_np = np.array(image)
    color = get_vehicle_color(img_np)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)

    vehicle = labels[idx.item()]
    confidence = confidence.item() * 100

    return {
        "vehicle": vehicle,
        "category": category_map[vehicle],
        "color": color,
        "confidence": round(confidence, 2)
    }
