import os
import random
import numpy as np  
import torch

def data_distribution(dataset, class_id, K, partition, n_parties, beta, seed, save_path):
    np.random.seed(seed)
    random.seed(seed)

    n_train = len(dataset)
    y_train = np.array([dataset[i][1] for i in range(n_train)])

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10

        N = n_train
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])

        times=[0 for i in range(K)]
        contain=[]
        for i in range(n_parties):
            current=[i%K]
            times[i%K]+=1
            j=1
            while (j<num):
                ind=random.randint(0,K-1)
                if (ind not in current):
                    j=j+1
                    current.append(ind)
                    times[ind]+=1
            contain.append(current)
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
        for i in range(K):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k,times[i])
            ids=0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                    ids+=1
        for i in range(n_parties):
            net_dataidx_map[i] = net_dataidx_map[i].tolist()

    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

        for i in range(n_parties):
            net_dataidx_map[i] = net_dataidx_map[i].tolist()

    # Save the partitioned datasets as .pt files
    os.makedirs(save_path + 'client_datasets', exist_ok=True)
    for party in range(n_parties):
        party_data = [dataset[idx] for idx in net_dataidx_map[party]]
        torch.save(party_data, os.path.join(save_path, f'client_datasets/client_{party}_dataset.pt'))

    return net_dataidx_map