import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def calculate_accuracy_and_loss(loader, model, criterion):
  model.eval()

  total = 0
  correct = 0
  running_loss = 0.0
  with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # To calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # To calculate loss
        loss = criterion(outputs, labels)  # Accumulate testing loss
        running_loss += loss.item()

  accuracy = 100 * correct / total
  loss = running_loss / len(loader)
  return accuracy, loss