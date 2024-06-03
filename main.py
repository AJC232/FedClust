import os
import time
import torch
from dotenv import load_dotenv
from get_args import getArgs
from server import run_server
from client import run_client

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device :",device)

os.makedirs('./models', exist_ok=True)
os.makedirs('./local_models', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

def main():
    args = getArgs()

    if args.type == 'server':
        start = time.time()

        run_server()

        end = time.time()
        total_time = (end - start)/60
        print(f"Total time: {total_time} minutes")
    
    else:
        start = time.time()

        run_client(args.client_id)

        end = time.time()
        total_time = (end - start)/60
        print(f"Total time: {total_time} minutes")

if __name__ == "__main__":
    main()