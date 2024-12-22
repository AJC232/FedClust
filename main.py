import os
import time
import torch
import threading
from arguments import getArgs
from server import run_server
from client import run_client

# Parse command-line arguments and store them in args
args = getArgs()

# Create a directory for the experiment if it doesn't already exist
os.makedirs(args.drive_path + args.exp_name, exist_ok=True)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using Device : ", device)

# Create and start a new thread to run the server
server_thread = threading.Thread(target=run_server)
server_thread.start()

# Pause the main thread for 60 seconds to allow the server to initialize
time.sleep(60)

# Initialize an empty list to hold client threads
client_threads = []

# Create and start a new thread for each client
for i in range(args.clients):
    client_thread = threading.Thread(target=run_client, args=(i,))
    client_thread.start()
    # Add the client thread to the list of client threads
    client_threads.append(client_thread)

# Wait for the server thread to finish
server_thread.join()

# Wait for all client threads to finish
for client_thread in client_threads:
    client_thread.join()
