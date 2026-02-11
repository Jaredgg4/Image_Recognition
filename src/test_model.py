# test_model.py
import torch
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request

model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open("src/assets/astronaut2.jpg")
img_tensor = transform(img).unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(img_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

print("Top 5 predictions:")
for i in range(5):
    print(f"{top5_catid[i].item()}: {top5_prob[i].item():.4f}")

url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
with urllib.request.urlopen(url) as f:
    labels = json.load(f)