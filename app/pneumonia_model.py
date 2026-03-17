import torch
import torchvision.transforms as transforms
from PIL import Image

model = torch.load("model/pneumonia_model.pt", map_location="cpu")
model.eval()

classes = ["normal","pneumonia"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_pneumonia(img: Image.Image):

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

    confidence, pred = torch.max(probs,1)

    prediction = classes[pred.item()]

    all_probs = {
        classes[i]: float(probs[0][i])
        for i in range(len(classes))
    }

    return prediction, confidence.item(), all_probs