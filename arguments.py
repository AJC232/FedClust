import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive_path', type=str, required=True, help='write root path')
    parser.add_argument('--exp_name', type=str, required=True, help='write experiment name')
    parser.add_argument('--clients', type=int, required=True, help='enter number of clients')
    parser.add_argument('--distribution', type=str, required=True, help='enter type of distribution')
    parser.add_argument('--rounds', type=int, required=True, help='enter number of rounds')
    parser.add_argument('--clusters', type=int, required=True, help='enter number of clusters')
    parser.add_argument('--epochs', type=int, required=True, help='enter number of epochs')
    parser.add_argument('--dataset', type=str, required=True, help='enter type of dataset')
 
    args = parser.parse_args()
    return args
