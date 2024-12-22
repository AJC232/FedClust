import os
from communication import Communication
import threading
import csv
import torch
from model import MOON, resnet10
import main_datasets
from update_global_model import update_global_model
from clusters import create_clusters
from unbiased import create_unbiased_model
from accuracy import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score
from data_distribution import data_distribution
from arguments import getArgs

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments and store them in args
args = getArgs()

# Set the file name for logging
file_name = args.drive_path + args.exp_name + 'logs/' + args.dataset + '_log'

# Create directories for saving models and logs
os.makedirs(args.drive_path + args.exp_name + 'models', exist_ok=True)

# Set the paths for saving models and logs
global_model_path = args.drive_path + args.exp_name + 'models/global_model.pth'
global_optimizer_path = args.drive_path + args.exp_name + 'models/global_optimizer.pth'
model_clusters_path = args.drive_path + args.exp_name + 'models/model_clusters.pth'
unbiased_model_path = args.drive_path + args.exp_name + 'models/unbiased_model.pth'

# Function to process the client
def process_client(conn, comm):
    print("sending signal to client...")
    comm.send_signal("Start training", conn)
    print("signal sent to client")

    print("waiting...")
    signal = comm.receive_signal(conn)
    print(f"signal: {signal} received")

# Function to run the server
def run_server():
  # Create the Communication object
  comm = Communication(host='127.0.0.1', port=9999)

  # Open a CSV file to store the results
  with open(file_name + '_server.csv', 'w', newline='') as csvfile:
    # Define the fieldnames for the CSV file
    fieldnames = ['round', 'training_accuracy', 'testing_accuracy', 'precision', 'recall', 'f1-score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

  trainset = ""
  testset = ""
 
  # Load the training and testing datasets
  if args.dataset == 'mnist':
    trainset = main_datasets.load_mnist_dataset(train=True)
    testset = main_datasets.load_mnist_dataset(train=False)
  elif args.dataset == 'fmnist':
    trainset = main_datasets.load_fmnist_dataset(train=True)
    testset = main_datasets.load_fmnist_dataset(train=False)
  elif args.dataset == 'cifar10':
    trainset = main_datasets.load_cifar10_dataset(train=True)
    testset = main_datasets.load_cifar10_dataset(train=False)
  elif args.dataset == 'svhn':
    trainset = main_datasets.load_svhn_dataset(train=True)
    testset = main_datasets.load_svhn_dataset(train=False)

  # Create the DataLoader objects for the training and testing datasets
  trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
  testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)
  
  # Initialize the MOON model, model_clusters, and unbiased_model
  global_model = MOON().to(device)
  model_clusters = {}
  unbiased_model = MOON().to(device)
  # global_model = resnet10().to(device)
  # model_clusters = {}
  # unbiased_model = resnet10().to(device)

  # datasets = data_distribution(trainset, Num_of_clients, 0.5)
  # trainset_np = dataset_to_numpy(trainset)
  
  # Distribute the data among the clients
  datasets = data_distribution(trainset, -1, 10, args.distribution, args.clients, 0.5, 42, args.drive_path + args.exp_name)
  data_sizes = [len(dataset) for dataset in datasets.values()]

  # Initialize the Communication object
  comm.init_server()
  print("--------------------------------------------------------------server running--------------------------------------------------------------")

  # Create a list to store the connections
  connections = []
  for i in range(args.clients):
    conn = comm.server_accept()
    connections.append(conn)

  # Start the training process
  for round in range(args.rounds):
    local_models = []
    threads = []
    global_model.train()

    # Save the global model, model_clusters, and unbiased_model
    torch.save(global_model.state_dict(), global_model_path)
    torch.save(model_clusters, model_clusters_path)
    torch.save(unbiased_model.state_dict(), unbiased_model_path)

    # for i in range(Num_of_clients):
    
    # Create a thread for each client
    for conn in connections:
      # conn = comm.server_accept()
      # thread = threading.Thread(target=process_client, args=(conn, comm, global_model, local_models))
      thread = threading.Thread(target=process_client, args=(conn, comm))
      thread.start()
      threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
      thread.join()

    # Load the local models
    for i in range(args.clients):
      local_model = MOON().to(device)
      # local_model = resnet10().to(device)
      local_model.load_state_dict(torch.load(args.drive_path + args.exp_name + f'local_models/client_{i}_model.pt'))
      local_models.append(local_model)

    # Update the global model, model_clusters, and unbiased_model
    global_model = update_global_model(global_model, local_models, data_sizes)
    model_clusters = create_clusters(local_models, args.clusters, args.clustering_method)
    unbiased_model = create_unbiased_model(model_clusters)

    # Calculate the accuracy, precision, recall, and F1 score
    train_accuracy = calculate_accuracy(trainloader, global_model)
    test_accuracy = calculate_accuracy(testloader, global_model)
    precision = calculate_precision(testloader, local_model)
    recall = calculate_recall(testloader, local_model)
    f1_score = calculate_f1_score(precision, recall)

    # Print the results
    print("Train Accuracy on server: ", train_accuracy)
    print("Test Accuracy on server: ", test_accuracy)
    print(f"Precision on server: {precision:.4f}")
    print(f"Recall on server: {recall:.4f}")
    print(f"F1 Score on server: {f1_score:.4f}")

    # Log the results in the CSV file
    with open(file_name + '_server.csv', 'a', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames)
      writer.writerow({'round': round + 1, 'training_accuracy': train_accuracy, 
                       'testing_accuracy': test_accuracy, 'precision': precision, 'recall': recall, 
                       'f1-score': f1_score})

  # Calculate the final accuracy
  train_accuracy = calculate_accuracy(trainloader, global_model)
  test_accuracy = calculate_accuracy(testloader, global_model)
  print("Final Train Accuracy on server: ", train_accuracy)
  print("Final Test Accuracy on server: ", test_accuracy)

  # Close the connections
  for conn in connections:
    comm.close_connection(conn)
  
  # Close the server
  comm.close_server()
