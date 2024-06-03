import os
import csv
import torch
import threading
import datasets as ds
from model import MOON
from dotenv import load_dotenv
from communication import Communication
from accuracy import calculate_accuracy
from model_clusters import create_clusters
from global_model import update_global_model
from unbiased_model import create_unbiased_model
from data_distribution import data_distribution

load_dotenv()

def process_client(conn, comm):
    print("sending signal to client...")
    comm.send_signal("Start training", conn)
    print("signal sent to client")

    print("waiting...")
    signal = comm.receive_signal(conn)
    print(f"signal: {signal} received")
    # comm.close_connection(conn)

# Function to run the server
def run_server():
  device = os.getenv("DEVICE")
  # Num_of_clients = int(os.getenv("Num_of_clients"))
  # Rounds = int(os.getenv("Rounds"))
  Num_of_clients = 3
  Rounds = 10
  global_model_path = os.getenv("global_model_path")
  model_clusters_path = os.getenv("model_clusters_path")
  unbiased_model_path = os.getenv("unbiased_model_path")
  log_file_path = os.getenv("log_file_path")
  log_file_name = os.getenv("log_file_name")

  comm = Communication(host='10.100.64.63', port=9999)

  with open(log_file_path + log_file_name + '_server.csv', 'w', newline='') as csvfile:
    fieldnames = ['round', 'training_accuracy', 'testing_accuracy']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

  trainset = ds.load_mnist_dataset(train=True)
  trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
  testset = ds.load_mnist_dataset(train=False)
  testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)

  # usps_trainset = load_usps_dataset(train=True)
  # usps_trainloader = torch.utils.data.DataLoader(dataset=usps_trainset, batch_size=64, shuffle=True)
  # usps_testset = load_usps_dataset(train=False)
  # usps_testloader = torch.utils.data.DataLoader(dataset=usps_testset, batch_size=64, shuffle=False)

  # trainset = ds.load_cifar10_dataset(train=True)
  # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
  # testset = ds.load_cifar10_dataset(train=False)
  # testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)

  # trainset = load_svhn_dataset(train=True)
  # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
  # testset = load_svhn_dataset(train=False)
  # testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)

  global_model = MOON().to(device)
  model_clusters = {}
  unbiased_model = MOON().to(device)

  datasets = data_distribution(trainset, Num_of_clients, 0.5)
  data_sizes = [len(dataset) for dataset in datasets.values()]

  comm.init_server()
  print("----------------------------------------------------server running--------------------------------------------------------------")

  connections = []
  for i in range(Num_of_clients):
    conn = comm.server_accept()
    connections.append(conn)

  for round in range(Rounds):
    local_models = []
    threads = []
    global_model.train()

    torch.save(global_model.state_dict(), global_model_path)
    torch.save(model_clusters, model_clusters_path)
    torch.save(unbiased_model.state_dict(), unbiased_model_path)

    # for i in range(Num_of_clients):
    print(connections)
    for conn in connections:
      # conn = comm.server_accept()
      thread = threading.Thread(target=process_client, args=(conn, comm))
      thread.start()
      threads.append(thread)
      print("------", len(threads))

    # while len(threads) != Num_of_clients:
    #   pass

    for thread in threads:
      thread.join()
      print("------", len(threads))

    for i in range(Num_of_clients):
      local_model = MOON().to(device)
      local_model.load_state_dict(torch.load(f'./local_models/client_{i}_model.pt'))
      local_models.append(local_model)

    global_model = update_global_model(global_model, local_models, data_sizes)
    model_clusters = create_clusters(local_models, 2)
    unbiased_model = create_unbiased_model(model_clusters)

    train_accuracy = calculate_accuracy(trainloader, global_model)
    test_accuracy = calculate_accuracy(testloader, global_model)
    print("MNIST Train Accuracy on server: ", train_accuracy)
    print("MNIST Test Accuracy on server: ", test_accuracy)

    # usps_train_accuracy = calculate_accuracy(usps_trainloader, global_model)
    # usps_test_accuracy = calculate_accuracy(usps_testloader, global_model)
    # print("USPS Train Accuracy on server: ", usps_train_accuracy)
    # print("USPS Test Accuracy on server: ", usps_test_accuracy)

    # train_accuracy = calculate_accuracy(trainloader, global_model)
    # test_accuracy = calculate_accuracy(testloader, global_model)
    # print("CIFAR10 Train Accuracy on server: ", train_accuracy)
    # print("CIFAR10 Test Accuracy on server: ", test_accuracy)

    with open(log_file_path + log_file_name + '_server.csv', 'a', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames)
      writer.writerow({'round': round + 1, 'training_accuracy': train_accuracy, 'testing_accuracy': test_accuracy})

  train_accuracy = calculate_accuracy(trainloader, global_model)
  test_accuracy = calculate_accuracy(testloader, global_model)
  print("Final MNIST Train Accuracy on server: ", train_accuracy)
  print("Final MNIST Test Accuracy on server: ", test_accuracy)
  # print("Final CIFAR10 Train Accuracy on server: ", train_accuracy)
  # print("Final CIFAR10 Test Accuracy on server: ", test_accuracy)

  for conn in connections:
    comm.close_connection(conn)
  comm.close_server()