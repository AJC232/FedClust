import torch

# Function to compute the cosine similarity between two vectors
def cosine_similarity(v1, v2):
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
    