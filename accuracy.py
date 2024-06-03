import os
import torch
from dotenv import load_dotenv

load_dotenv()

device = os.getenv("DEVICE")

def calculate_accuracy(loader, model):
  model.eval()

  total = 0
  correct = 0
  with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # To calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  return accuracy