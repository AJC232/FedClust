import os
import csv
import torch
import torch.optim as optim
import datasets as ds
from model import MOON
from loss import Loss
from dotenv import load_dotenv
from model_clusters import create_positive_cluster
from accuracy import calculate_accuracy
from communication import Communication

load_dotenv()

# Function to run the client
def run_client(client_id):
  device = os.getenv("DEVICE")
  # Rounds = int(os.getenv("Rounds"))
  # Epochs = int(os.getenv("Epochs"))
  Rounds = 10
  Epochs = 5
  global_model_path = os.getenv("global_model_path")
  model_clusters_path = os.getenv("model_clusters_path")
  unbiased_model_path = os.getenv("unbiased_model_path")
  local_models_path = os.getenv("local_models_path")

  print(f"Rounds: {Rounds}, Epochs: {Epochs}")

  comm = Communication(host='10.100.64.63', port=9999)
  print(f"----------------------------------------------------client_{client_id} running--------------------------------------------------------------")

  log_file_path = os.getenv('log_file_path')
  log_file_name = os.getenv('log_file_name')
  with open(log_file_path + log_file_name + '_client_' + str(client_id) + '.csv', 'w', newline='') as csvfile:
    fieldnames = ['round', 'training_accuracy', 'testing_accuracy']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

  # trainset = []
  # testset = []
  # if client_id == 1:
  #   trainset = load_mnist_dataset(train=True)
  #   testset = load_mnist_dataset(train=False)
  # else:
  #   trainset = load_usps_dataset(train=True)
  #   testset = load_usps_dataset(train=False)

  trainset = torch.load(f'./datasets/client_{client_id}_dataset.pt')
  testset = ds.load_mnist_dataset(train=False)
  # testset = ds.load_cifar10_dataset(train=False)
  # testset = load_svhn_dataset(train=False)
  print(f'Cliet {client_id} dataset size: ', len(trainset))
  trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
  testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)

  comm.init_client(port=9999)

  for round in range(Rounds):
    print("Round", round+1, "started...")
    # comm.init_client(port=9999)
    # # Receive the global model from the server
    # global_model = comm.receive_model(comm.client)
    # global_model = global_model.to(device)
    # print("global model recieved")

    # Receive the global model from the server
    signal = comm.receive_signal(comm.client)
    print("recieved signal: ", signal)

    global_model = MOON().to(device)
    global_model.load_state_dict(torch.load(global_model_path))
    global_model = global_model.to(device)

    model_clusters = torch.load(model_clusters_path)
    # print("Type of model_clusters: ", type(model_clusters))

    unbiased_model = MOON().to(device)
    unbiased_model.load_state_dict(torch.load(unbiased_model_path))

    #Make a copy of global model
    local_model = MOON().to(device)
    local_model.load_state_dict(global_model.state_dict())

    # Define a loss function and optimizer
    criterion = Loss()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.00001)
    # if os.path.exists(global_optimizer_path):
    #   optimizer.load_state_dict(torch.load(global_optimizer_path))

    for epoch in range(Epochs):
      # print("Epoch", epoch+1, "started...")
      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        local_outputs = local_model(inputs)
        local_rep = local_model.Rw(inputs)
        positive_cluster, negative_clusters = create_positive_cluster(model_clusters, local_model)
        # positive_reps = []
        # negative_reps = []
        # for model in positive_cluster:
        #   positive_reps.append(model.Rw(inputs))
        # for cluster in negative_clusters.values():
        #   for model in cluster:
        #     negative_reps.append(model.Rw(inputs))
        # unbiased_rep = unbiased_model.Rw(inputs)

        # loss = criterion(local_outputs, labels)
        # print(type(local_outputs))
        # print(type(labels))
        # print(type(local_rep))
        # print(type(positive_cluster))
        # print(type(negative_clusters))
        # print(type(unbiased_rep))
        loss = criterion(local_outputs, labels, local_model, positive_cluster, negative_clusters, unbiased_model)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Print loss for this batch
        running_loss += loss.item()

      print('Epoch %d, Loss: %.3f' % (epoch + 1, running_loss/len(trainloader)))


    print('Finished Training')

    train_accuracy = calculate_accuracy(trainloader, local_model)
    test_accuracy = calculate_accuracy(testloader, local_model)

    with open(log_file_path + log_file_name + '_client_' + str(client_id) + '.csv', 'a', newline='') as csvfile:
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