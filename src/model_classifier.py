import torch
from torchvision import models, transforms
from PIL import Image
import json

class ImageClassifier:
    def __init__(self, model_name='resnet50'):
        """
        Initialize the classifier
        Args:
            model_name: 'resnet50', 'resnet101', or 'vit_b_16'
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=True)
        elif model_name == 'vit_b_16':
            self.model = models.vit_b_16(pretrained=True)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load labels
        self.labels = self._load_labels()
    
    def _load_labels(self):
        """Load ImageNet class labels"""
        return None
    
    def predict(self, image, top_k=5):
        """
        Classify an image
        Args:
            image: PIL Image or path to image
            top_k: Number of top predictions to return
        Returns:
            List of tuples (class_name, probability)
        """

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, top_k)
        
        results = []
        for i in range(top_k):
            class_id = top_catid[i].item()
            prob = top_prob[i].item()
            class_name = self.labels[class_id] if self.labels else f"Class {class_id}"
            results.append((class_name, prob))
        
        return results