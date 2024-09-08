from communication import Communication
import os
import csv
import torch
from model import MOON, resnet10
from accuracy import calculate_accuracy
from arguments import getArgs
import main_datasets
from loss import Loss
import torch.optim as optim
from clusters import create_positive_cluster

args = getArgs()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_name = args.drive_path + args.exp_name + 'logs/' + args.dataset + '_log'

os.makedirs(args.drive_path + args.exp_name + 'models', exist_ok=True)
global_model_path = args.drive_path + args.exp_name + 'models/global_model.pth'
global_optimizer_path = args.drive_path + args.exp_name + 'models/global_optimizer.pth'
model_clusters_path = args.drive_path + args.exp_name + 'models/model_clusters.pth'
unbiased_model_path = args.drive_path + args.exp_name + 'models/unbiased_model.pth'

local_models_path = args.drive_path + args.exp_name + 'local_models/'
os.makedirs(local_models_path, exist_ok=True)
os.makedirs(args.drive_path + args.exp_name + 'logs', exist_ok=True)

# Function to run the client
def run_client(client_id):
  comm = Communication(host='127.0.0.1', port=9999)
  print(f"--------------------------------------------------------------client_{client_id} running--------------------------------------------------------------")

  with open(file_name + '_client_' + str(client_id) + '.csv', 'w', newline='') as csvfile:
    fieldnames = ['round', 'training_accuracy', 'testing_accuracy']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

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
  trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
  testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)

  comm.init_client(port=9999)

  for round in range(args.rounds):
    print("Round", round+1, "started...")

    # Receive the global model from the server
    signal = comm.receive_signal(comm.client)
    print("recieved signal: ", signal)

    global_model = MOON().to(device)
    # global_model = resnet10().to(device)
    global_model.load_state_dict(torch.load(global_model_path))
    global_model = global_model.to(device)

    model_clusters = torch.load(model_clusters_path)
    # print("Type of model_clusters: ", type(model_clusters))

    unbiased_model = MOON().to(device)
    # unbiased_model = resnet10().to(device)
    unbiased_model.load_state_dict(torch.load(unbiased_model_path))

    #Make a copy of global model
    local_model = MOON().to(device)
    # local_model = resnet10().to(device)
    local_model.load_state_dict(global_model.state_dict())

    # Define a loss function and optimizer
    criterion = Loss()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.00001)
    # if os.path.exists(global_optimizer_path):
    #   optimizer.load_state_dict(torch.load(global_optimizer_path))

    for epoch in range(args.epochs):
      # print("Epoch", epoch+1, "started...")
      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # print(f"Inputs shape: {inputs.shape}, Targets shape: {labels.shape}")

        optimizer.zero_grad()

        local_outputs = local_model(inputs)
        local_rep = local_model.Rw(inputs)
        positive_cluster, negative_clusters = create_positive_cluster(model_clusters, local_model)
        
        loss = criterion(local_outputs, labels, local_model, positive_cluster, negative_clusters, unbiased_model)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Print loss for this batch
        running_loss += loss.item()

      # print('Epoch %d, Loss: %.3f' % (epoch + 1, running_loss/len(trainloader)))


    print('Finished Training')

    train_accuracy = calculate_accuracy(trainloader, local_model)
    test_accuracy = calculate_accuracy(testloader, local_model)

    with open(file_name + '_client_' + str(client_id) + '.csv', 'a', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames)
      writer.writerow({'round': round + 1, 'training_accuracy': train_accuracy, 'testing_accuracy': test_accuracy})

    print(f"Train Accuracy on client {client_id}: ", train_accuracy)
    print(f"Test Accuracy on client {client_id}: ", test_accuracy)

    local_model.train()

    # print("local model hash on client: ", hash_model_parameters(local_model))

    torch.save(local_model.state_dict(), local_models_path + f'client_{client_id}_model.pt')
    # torch.save(optimizer.state_dict(), global_optimizer_path)

    comm.send_signal(f"Round {round+1} completed", comm.client)

  comm.close_client()
    # print("sending model to server...")
    # comm.send_model(local_model, comm.client)
    # print("model sent to server")