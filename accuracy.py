import torch

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to calculate accuracy of the model
def calculate_accuracy(loader, model):
    # Set the model to evaluation mode
    model.eval()

    # Initialize total and correct counters
    total = 0
    correct = 0

    # Disable gradient calculation for efficiency
    with torch.no_grad():
        # Iterate over the data loader
        for images, labels in loader:
            # Move images and labels to the appropriate device
            images, labels = images.to(device), labels.to(device)
            # Get model predictions
            outputs = model(images)
            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs, 1)

            # Update total and correct counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy as a percentage
    accuracy = 100 * correct / total
    return accuracy

# Function to calculate precision of the model
def calculate_precision(dataloader, model):
    # Set the model to evaluation mode
    model.eval()
    # Initialize true positives and false positives counters
    true_positives = 0
    false_positives = 0

    # Disable gradient calculation for efficiency
    with torch.no_grad():
        # Iterate over the data loader
        for inputs, labels in dataloader:
            # Move inputs and labels to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            # Get the predicted class with the highest score
            predictions = torch.argmax(outputs, dim=1)
            
            # Update true positives and false positives counters
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
    
    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    return precision

# Function to calculate recall of the model
def calculate_recall(dataloader, model):
    # Set the model to evaluation mode
    model.eval() 
    # Initialize true positives and false negatives counters
    true_positives = 0
    false_negatives = 0

    # Disable gradient calculation for efficiency
    with torch.no_grad():  
        # Iterate over the data loader
        for inputs, labels in dataloader:
            # Move inputs and labels to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            # Get the predicted class with the highest score
            predictions = torch.argmax(outputs, dim=1)
            
            # Update true positives and false negatives counters
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()
    
    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    return recall

# Function to calculate F1 score of the model
def calculate_f1_score(precision, recall):
    # Calculate F1 score
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
