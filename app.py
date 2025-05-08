from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

class_names = ["notumor", "pituitary", "meningioma", "glioma"]
idx_to_class = {i: name for i, name in enumerate(class_names)}

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model():
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    
    image_tensor = test_transforms(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
    result = {idx_to_class[i]: float(probabilities[0][i]) for i in range(len(class_names))}
    
    return jsonify(result)

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify(class_names)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
