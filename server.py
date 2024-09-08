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
from accuracy import calculate_accuracy
from data_distribution import data_distribution
from arguments import getArgs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = getArgs()

file_name = args.drive_path + args.exp_name + 'logs/' + args.dataset + '_log'

os.makedirs(args.drive_path + args.exp_name + 'models', exist_ok=True)
global_model_path = args.drive_path + args.exp_name + 'models/global_model.pth'
global_optimizer_path = args.drive_path + args.exp_name + 'models/global_optimizer.pth'
model_clusters_path = args.drive_path + args.exp_name + 'models/model_clusters.pth'
unbiased_model_path = args.drive_path + args.exp_name + 'models/unbiased_model.pth'

def process_client(conn, comm):
    print("sending signal to client...")
    comm.send_signal("Start training", conn)
    print("signal sent to client")

    print("waiting...")
    signal = comm.receive_signal(conn)
    print(f"signal: {signal} received")

# Function to run the server
def run_server():
  comm = Communication(host='127.0.0.1', port=9999)

  with open(file_name + '_server.csv', 'w', newline='') as csvfile:
    fieldnames = ['round', 'training_accuracy', 'testing_accuracy']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

  trainset = ""
  testset = ""
 
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

  trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
  testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)
  

  global_model = MOON().to(device)
  model_clusters = {}
  unbiased_model = MOON().to(device)
  # global_model = resnet10().to(device)
  # model_clusters = {}
  # unbiased_model = resnet10().to(device)

  # datasets = data_distribution(trainset, Num_of_clients, 0.5)
  # trainset_np = dataset_to_numpy(trainset)
  datasets = data_distribution(trainset, -1, 10, args.distribution, args.clients, 0.5, 42, args.drive_path + args.exp_name)
  data_sizes = [len(dataset) for dataset in datasets.values()]

  comm.init_server()
  print("--------------------------------------------------------------server running--------------------------------------------------------------")

  connections = []
  for i in range(args.clients):
    conn = comm.server_accept()
    connections.append(conn)

  for round in range(args.rounds):
    local_models = []
    threads = []
    global_model.train()

    torch.save(global_model.state_dict(), global_model_path)
    torch.save(model_clusters, model_clusters_path)
    torch.save(unbiased_model.state_dict(), unbiased_model_path)

    # for i in range(Num_of_clients):
    for conn in connections:
      # conn = comm.server_accept()
      # thread = threading.Thread(target=process_client, args=(conn, comm, global_model, local_models))
      thread = threading.Thread(target=process_client, args=(conn, comm))
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

    for i in range(args.clients):
      local_model = MOON().to(device)
      # local_model = resnet10().to(device)
      local_model.load_state_dict(torch.load(args.drive_path + args.exp_name + f'local_models/client_{i}_model.pt'))
      local_models.append(local_model)

    global_model = update_global_model(global_model, local_models, data_sizes)
    model_clusters = create_clusters(local_models, args.clusters)
    unbiased_model = create_unbiased_model(model_clusters)

    train_accuracy = calculate_accuracy(trainloader, global_model)
    test_accuracy = calculate_accuracy(testloader, global_model)
    print("Train Accuracy on server: ", train_accuracy)
    print("Test Accuracy on server: ", test_accuracy)

    with open(file_name + '_server.csv', 'a', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames)
      writer.writerow({'round': round + 1, 'training_accuracy': train_accuracy, 'testing_accuracy': test_accuracy})

  train_accuracy = calculate_accuracy(trainloader, global_model)
  test_accuracy = calculate_accuracy(testloader, global_model)
  print("Final Train Accuracy on server: ", train_accuracy)
  print("Final Test Accuracy on server: ", test_accuracy)

  for conn in connections:
    comm.close_connection(conn)
  comm.close_server()