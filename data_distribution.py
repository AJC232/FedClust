import os
import torch
import numpy as np
from torch.utils.data import Subset

def data_distribution(dataset, num_clients, alpha):
    """
    Distribute data among clients using Dirichlet distribution.

    Parameters:
    dataset (torch.utils.data.Dataset): The MNIST dataset to be distributed.
    num_clients (int): The number of clients.
    alpha (float): The concentration parameter for the Dirichlet distribution.

    Returns:
    dict: A dictionary where the keys are the client IDs and the values are the data for each client.
    """

    # Number of data points
    num_data = len(dataset)

    # Create a distribution for the number of data points per client
    distribution = np.random.dirichlet(np.repeat(alpha, num_clients))

    # Allocate data points to clients
    data_per_client = np.random.multinomial(num_data, distribution)

    # Distribute the data
    data_distribution = {}
    data_start = 0
    for i in range(num_clients):
        data_end = data_start + data_per_client[i]
        data_distribution[i] = Subset(dataset, range(data_start, data_end))
        data_start = data_end

    os.makedirs('./datasets', exist_ok=True)
    for i, dataset in data_distribution.items():
        torch.save(list(dataset), f'./datasets/client_{i}_dataset.pt')

    return data_distribution