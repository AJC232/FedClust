import torch

def extract_model_weights(model):
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().flatten())
    return torch.cat(weights)