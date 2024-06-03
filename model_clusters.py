import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
# from sklearn.metrics.pairwise import cosine_similarity
from cosine_similarity import cosine_similarity

def extract_model_weights(model):
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().flatten())
    return torch.cat(weights)


# Function to perform k-means clustering with cosine similarity
def create_clusters(models, n_clusters):
    # Extract weights from each model and convert them to numpy arrays for sklearn functions
    model_weights = [extract_model_weights(model).cpu().numpy() for model in models]

    # Normalize the weights to unit vectors for cosine similarity
    model_weights_normalized = normalize(model_weights)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++')
    kmeans.fit(model_weights_normalized)

    # Get the cluster labels
    labels = kmeans.labels_

    # Group models by their cluster labels
    clusters = {i: [] for i in range(n_clusters)}
    for model, label in zip(models, labels):
        clusters[label].append(model)

    return clusters

def create_positive_cluster(clusters, local_model):
    # Extract and normalize weights of the local model
    local_weights = extract_model_weights(local_model)
    # local_weights_normalized = normalize([local_weights])[0]

    min_sim = float('inf')
    positive_cluster_key = None

    # Iterate through each cluster and each model within those clusters
    for cluster_key, cluster in clusters.items():
        for model in cluster:
            # Extract and normalize weights of the current model
            model_weights = extract_model_weights(model)
            # model_weights_normalized = normalize([model_weights])[0]

            # Calculate cosine similarity
            sim = cosine_similarity(local_weights, model_weights)

            # Update the minimum similarity and corresponding cluster if necessary
            if sim < min_sim:
                min_sim = sim
                positive_cluster_key = cluster_key

    positive_cluster = clusters.pop(positive_cluster_key, {})

    return positive_cluster, clusters
