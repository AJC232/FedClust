import os
import time
import torch
import threading
from arguments import getArgs
from server import run_server
from client import run_client

args = getArgs()
os.makedirs(args.drive_path + args.exp_name, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using Device : ", device)

server_thread = threading.Thread(target=run_server)
server_thread.start()

time.sleep(60)

client_threads = []
for i in range(args.clients):
  client_thread = threading.Thread(target=run_client, args=(i,))
  client_thread.start()
  client_threads.append(client_thread)

server_thread.join()
for client_thread in client_threads:
  client_thread.join()