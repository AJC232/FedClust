import torch

# Function to extract model weights
def extract_model_weights(model):
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().flatten())
    return torch.cat(weights)