from communication import Communication
import os
import csv
import torch
from model import MOON, resnet10
from accuracy import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score
from arguments import getArgs
import main_datasets
from loss import Loss
import torch.optim as optim
from clusters import create_positive_cluster

# Parse command-line arguments and store them in args
args = getArgs()

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the file name for logging
file_name = args.drive_path + args.exp_name + 'logs/' + args.dataset + '_log'

# Create directories for saving models and logs
os.makedirs(args.drive_path + args.exp_name + 'models', exist_ok=True)
os.makedirs(local_models_path, exist_ok=True)
os.makedirs(args.drive_path + args.exp_name + 'logs', exist_ok=True)

# Set the paths for saving models and logs 
global_model_path = args.drive_path + args.exp_name + 'models/global_model.pth'
global_optimizer_path = args.drive_path + args.exp_name + 'models/global_optimizer.pth'
model_clusters_path = args.drive_path + args.exp_name + 'models/model_clusters.pth'
unbiased_model_path = args.drive_path + args.exp_name + 'models/unbiased_model.pth'
local_models_path = args.drive_path + args.exp_name + 'local_models/'

# Function to run the client
def run_client(client_id):
  # Create the Communication object
  comm = Communication(host='127.0.0.1', port=9999)
  
  # Print a message to indicate that the client is running 
  print(f"--------------------------------------------------------------client_{client_id} running--------------------------------------------------------------")

  # Create a CSV file to store the results
  with open(file_name + '_client_' + str(client_id) + '.csv', 'w', newline='') as csvfile:
    # Define the fieldnames for the CSV file
    fieldnames = ['round', 'training_accuracy', 'testing_accuracy', 'precision', 'recall', 'f1-score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

  # Load the training and testing datasets
  trainset = torch.load(args.drive_path + args.exp_name + f'client_datasets/client_{client_id}_dataset.pt')
  testset = ""
  if args.dataset == 'mnist':
    testset = main_datasets.load_mnist_dataset(train=False)
  elif args.dataset == 'fmnist':
    testset = main_datasets.load_fmnist_dataset(train=False)
  elif args.dataset == 'cifar10':
    testset = main_datasets.load_cifar10_dataset(train=False)
  elif args.dataset == 'svhn':
    testset = main_datasets.load_svhn_dataset(train=False)

  print(f'Client {client_id} dataset size: ', len(trainset))

  # Create data loaders for the training and testing datasets
  trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
  testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)

  # Initialize the Communication object
  comm.init_client(port=9999)

  # Start the training process
  for round in range(args.rounds):
    print("Round", round+1, "started...")

    # Receive signal from server
    signal = comm.receive_signal(comm.client)
    print("recieved signal: ", signal)

    # Load the global model
    global_model = MOON().to(device)
    global_model.load_state_dict(torch.load(global_model_path))
    global_model = global_model.to(device)

    # Load the model clusters
    model_clusters = torch.load(model_clusters_path)

    # Load the unbiased model  
    unbiased_model = MOON().to(device)
    unbiased_model.load_state_dict(torch.load(unbiased_model_path))

    #Make a copy of global model as local_model
    local_model = MOON().to(device)
    local_model.load_state_dict(global_model.state_dict())

    # Define a loss function and optimizer
    criterion = Loss()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.00001)

    # Start local training of the model
    for epoch in range(args.epochs):
      # print("Epoch", epoch+1, "started...")
      
      # Set running loss to 0
      running_loss = 0.0

      # Iterate over the training data
      for i, data in enumerate(trainloader, 0):
        # Get the inputs and labels
        inputs, labels = data        
        inputs, labels = inputs.to(device), labels.to(device)
        # print(f"Inputs shape: {inputs.shape}, Targets shape: {labels.shape}")

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        local_outputs = local_model(inputs)
        
        # Get the representation of the local_model 
        local_rep = local_model.Rw(inputs)

        # Create positive and negative clusters
        positive_cluster, negative_clusters = create_positive_cluster(model_clusters, local_model)
        
        # Calculate the loss
        loss = criterion(local_outputs, labels, local_model, positive_cluster, negative_clusters, unbiased_model)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Calculate the running loss
        running_loss += loss.item()

      # print('Epoch %d, Loss: %.3f' % (epoch + 1, running_loss/len(trainloader)))


    print('Finished Training')

    # Calculate the training and testing accuracy
    train_accuracy = calculate_accuracy(trainloader, local_model)
    test_accuracy = calculate_accuracy(testloader, local_model)

    # Calculate the precision, recall, and F1 score
    precision = calculate_precision(testloader, local_model)
    recall = calculate_recall(testloader, local_model)
    f1_score = calculate_f1_score(precision, recall)

    # Log the results in the CSV file
    with open(file_name + '_client_' + str(client_id) + '.csv', 'a', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames)
      writer.writerow({'round': round + 1, 'training_accuracy': train_accuracy, 
                       'testing_accuracy': test_accuracy, 'precision': precision, 'recall': recall, 
                       'f1-score': f1_score})

    # Print the results
    print(f"Train Accuracy on client {client_id}: ", train_accuracy)
    print(f"Test Accuracy on client {client_id}: ", test_accuracy)
    print(f"Precision on client {client_id} {precision:.4f}")
    print(f"Recall on client {client_id}: {recall:.4f}")
    print(f"F1 Score on client {client_id}: {f1_score:.4f}")

    # Set the local model to training mode
    local_model.train()

    # Save the local model
    torch.save(local_model.state_dict(), local_models_path + f'client_{client_id}_model.pt')

    # Send the signal to the server that the round is completed
    comm.send_signal(f"Round {round+1} completed", comm.client)

  # Close the client connection
  comm.close_client()
    # print("sending model to server...")
    # comm.send_model(local_model, comm.client)
    # print("model sent to server")