import torch

def cosine_similarity(v1, v2):
    """Compute the cosine similarity between two vectors."""
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))