import argparse

# Function to parse command-line arguments
def getArgs():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    
    # Add argument for the root path
    parser.add_argument('--drive_path', type=str, required=True, help='write root path')
    # Add argument for the experiment name
    parser.add_argument('--exp_name', type=str, required=True, help='write experiment name')
    # Add argument for the number of clients
    parser.add_argument('--clients', type=int, required=True, help='enter number of clients')
    # Add argument for the type of data distribution
    parser.add_argument('--distribution', type=str, required=True, help='enter type of distribution')
    # Add argument for the number of rounds
    parser.add_argument('--rounds', type=int, required=True, help='enter number of rounds')
    # Add argument for the number of clusters for KMeans
    parser.add_argument('--clusters', type=int, required=True, help='enter number of clusters')
    # Add argument for the number of epochs
    parser.add_argument('--epochs', type=int, required=True, help='enter number of epochs')
    # Add argument for the type of dataset
    parser.add_argument('--dataset', type=str, required=True, help='enter type of dataset')
    # Add argument for the type of clustering method
    parser.add_argument('--clustering_method', type=str, required=True, help='enter type of clustering method')
 
    # Parse the command-line arguments
    args = parser.parse_args()
    # Return the parsed arguments
    return args
