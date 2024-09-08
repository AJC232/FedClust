import torch
from model import MOON

def average_model_weights(models):
    """Compute the average weights of a list of models."""
    avg_state_dict = models[0].state_dict()  # Get the state_dict of the first model

    # Initialize the average state_dict with zeros
    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.zeros_like(avg_state_dict[key])

    # Sum up the weights
    for model in models:
        state_dict = model.state_dict()
        for key in state_dict.keys():
            avg_state_dict[key] += state_dict[key]

    # Divide by the number of models to get the average
    for key in avg_state_dict.keys():
        avg_state_dict[key] /= len(models)

    return avg_state_dict

def create_unbiased_model(clusters):
    """Compute the averaged model from clusters of models."""
    averaged_state_dicts = []

    # Compute the average weights for each cluster
    for cluster_id, models in clusters.items():
        avg_state_dict = average_model_weights(models)
        averaged_state_dicts.append(avg_state_dict)

    # Compute the overall average of the averaged weights from each cluster
    overall_avg_state_dict = averaged_state_dicts[0]
    for key in overall_avg_state_dict.keys():
        overall_avg_state_dict[key] = torch.zeros_like(overall_avg_state_dict[key])

    for avg_state_dict in averaged_state_dicts:
        for key in avg_state_dict.keys():
            overall_avg_state_dict[key] += avg_state_dict[key]

    for key in overall_avg_state_dict.keys():
        overall_avg_state_dict[key] /= len(averaged_state_dicts)

    # Create a new model and load the averaged state_dict
    averaged_model = MOON()
    averaged_model.load_state_dict(overall_avg_state_dict)

    return averaged_model