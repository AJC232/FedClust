import torch
import torch.nn as nn
# from sklearn.metrics.pairwise import cosine_similarity
from cosine_similarity import cosine_similarity
from model_clusters import extract_model_weights

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.supervisedLoss = nn.CrossEntropyLoss()

    def forward(self, local_output, labels, local_rep, positive_cluster, negative_clusters, unbiased_model):
        supervised_loss = self.supervisedLoss(local_output, labels)

        local_weights = extract_model_weights(local_rep)
        unbiased_weights = extract_model_weights(unbiased_model)

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

        cluster_contrastive_loss = log_sum_exp_negative - log_sum_exp_positive

        unbiased_loss = torch.norm(local_weights - unbiased_weights)**2

        loss = supervised_loss + cluster_contrastive_loss + unbiased_loss
        return loss