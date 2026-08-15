from pathlib import Path
import json
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
classes=json.loads((ROOT/"models/product_classifier_meta.json").read_text())["classes"]
model=models.resnet18(weights=None); model.fc=nn.Linear(512,10)
model.load_state_dict(torch.load(ROOT/"models/product_classifier.pt",map_location="cpu")); model.eval()
tf=transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def classify_product_image(image_path:str)->dict:
    x=tf(Image.open(image_path).convert("L")).unsqueeze(0)
    with torch.no_grad(): p=torch.softmax(model(x),dim=1)[0]
    i=int(p.argmax())
    return {"label":classes[i],"confidence":float(p[i])}
