import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Step 1: Define the Model Architecture (Unchanged) ---
# This class MUST be identical to the one used for training.
class HybridPestNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridPestNet, self).__init__()
        self.vit = models.vit_b_16(weights=None)
        self.vit.heads = nn.Identity()

        resnet = models.resnet50(weights=None)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.custom_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 2048 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        vit_features = self.vit(x)
        resnet_features = self.resnet(x).view(x.size(0), -1)
        cnn_features = self.custom_cnn(x)
        combined_features = torch.cat((vit_features, resnet_features, cnn_features), dim=1)
        return self.classifier(combined_features)

# --- Step 2: Global Variables and Model Loading ---

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'pest_detection_hybrid_model_optimized.pth'
CLASS_NAMES = [
    'ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
    'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'
]
NUM_CLASSES = len(CLASS_NAMES)

# --- NEW SECTION 1: Load the recommendations database ---
# The server loads this file once at startup and keeps it in memory.
try:
    with open('recommendations.json', 'r', encoding='utf-8') as f:
        recommendations_db = json.load(f)
    print("✅ Recommendations database loaded successfully.")
except Exception as e:
    print(f"❌ Error loading recommendations.json: {e}")
    recommendations_db = {} # Use an empty dictionary as a fallback

# Load the trained model
model = HybridPestNet(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded on {DEVICE} and ready.")

# --- Step 3: Define the Prediction Function (Unchanged) ---

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_tensor):
    image_tensor = image_tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, top_catid = torch.max(probabilities, 0)
        predicted_class = CLASS_NAMES[top_catid.item()]
        confidence_score = confidence.item()
    return predicted_class, confidence_score

# --- Step 4: Define the API Endpoint (MODIFIED) ---

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'no image provided'}), 400

    image_data = request.json['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)

    try:
        tensor = transform_image(image_bytes)
        predicted_class, confidence_score = get_prediction(tensor)

        # --- NEW SECTION 2: Look up the recommendation for the predicted pest ---
        # The .get() method safely retrieves the recommendation.
        # If the pest is not found, it provides a default message.
        recommendation = recommendations_db.get(
            predicted_class, 
            {"error": "No recommendation found for this pest."}
        )

        # --- NEW SECTION 3: Add the recommendation to the JSON response ---
        # The frontend will now receive the pest, confidence, AND the recommendation.
        return jsonify({
            'pest': predicted_class,
            'confidence': f"{confidence_score*100:.2f}%",
            'recommendation': recommendation
        })
    except Exception as e:
        return jsonify({'error': f'error processing image: {str(e)}'}), 500

# --- Step 5: Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

