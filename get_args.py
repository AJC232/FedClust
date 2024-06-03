import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='client', help='server or client')
    parser.add_argument('--client_id', type=int, default='1', help='enter client id')
    args = parser.parse_args()
    return args