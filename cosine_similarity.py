import torch

def cosine_similarity(v1, v2):
    """Compute the cosine similarity between two vectors."""
    # Convert numpy arrays to PyTorch tensors
    # v1 = torch.from_numpy(v1)
    # v2 = torch.from_numpy(v2)
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))