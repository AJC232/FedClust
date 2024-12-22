import torch
import torch.nn as nn
from cosine_similarity import cosine_similarity
from model_weights import extract_model_weights

# Class for the loss function
class Loss(nn.Module):
    # Initialize the loss function
    def __init__(self):
        super(Loss, self).__init__()
        self.supervisedLoss = nn.CrossEntropyLoss()

    # Forward pass of the loss function
    def forward(self, local_output, labels, local_rep, positive_cluster, negative_clusters, unbiased_model):
        # Calculate the supervised loss
        supervised_loss = self.supervisedLoss(local_output, labels)

        # Compute local weights
        local_weights = extract_model_weights(local_rep)

        # Compute unbiased weights
        unbiased_weights = extract_model_weights(unbiased_model)

        # Compute positive and negative cluster weights and similarities
        positive_weights = []
        negative_weights = []
        for model in positive_cluster:
          positive_weights.append(extract_model_weights(model))
        for cluster in negative_clusters.values():
          for model in cluster:
            negative_weights.append(extract_model_weights(model))

        positive_similarities = [cosine_similarity(local_weights, c) for c in positive_weights]
        negative_similarities = [cosine_similarity(local_weights, c) for c in negative_weights]

        # log_sum_exp_positive = torch.logsumexp(torch.stack(positive_similarities), dim=0)
        # log_sum_exp_negative = torch.logsumexp(torch.stack(negative_similarities), dim=0)
        # Check if positive_similarities is not empty before stacking
        if positive_similarities:
            log_sum_exp_positive = torch.logsumexp(torch.stack(positive_similarities), dim=0)
        else:
            log_sum_exp_positive = 0

        # Check if negative_similarities is not empty before stacking
        if negative_similarities:
            log_sum_exp_negative = torch.logsumexp(torch.stack(negative_similarities), dim=0)
        else:
            log_sum_exp_negative = 0

        # Compute the cluster contrastive loss
        cluster_contrastive_loss = log_sum_exp_negative - log_sum_exp_positive

        # Compute the unbiased loss
        unbiased_loss = torch.norm(local_weights - unbiased_weights)**2

        # Compute the total loss
        loss = supervised_loss + cluster_contrastive_loss + unbiased_loss
        # loss = supervised_loss + cluster_contrastive_loss
        return loss