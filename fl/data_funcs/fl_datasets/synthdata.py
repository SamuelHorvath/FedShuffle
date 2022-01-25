import numpy as np
import torch

# from ..fl_dataset import FLDataset
from torch.utils.data import Dataset


class SynthData(Dataset):
    """
    Synthetic data.
    As in Federated Optimization in Heterogenous Networks,
    https://arxiv.org/pdf/1812.06127.pdf
    Run notebooks/create_datasets.ipynb to generate data
    """
    def __init__(self, fl_dataset, client_id=None):
        self.fl_dataset = fl_dataset
        self.set_client(client_id)

    def set_client(self, index=None):
        fl = self.fl_dataset
        if index is None:
            self.client_id = None
            self.length = len(fl.y)
            self.x = fl.x
            self.y = fl.y
        else:
            if index < 0 or index >= fl.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.length = fl.n[index + 1] - fl.n[index]
            self.x = fl.x[fl.n[index]:fl.n[index + 1]]
            self.y = fl.y[fl.n[index]:fl.n[index + 1]]

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x, y

    def __len__(self):
        return self.length


class SynthDataFL:
    """
    Synthetic data.
    As in Federated Optimization in Heterogenous Networks,
    https://arxiv.org/pdf/1812.06127.pdf
    Run notebooks/create_datasets.ipynb to generate data
    """
    def __init__(self, root, dataset):
        self.num_clients = 30
        self.data = np.load(f'{root}/{dataset}.npz')
        self.x = torch.from_numpy(self.data['x']).float()
        self.y = torch.from_numpy(self.data['y'])
        self.n = self.data['n']
        self.length = len(self.y)

    def __len__(self):
        return self.length


# class SynthData(FLDataset):
#     """
#     Synthetic data.
#     As in Federated Optimization in Heterogenous Networks,
#     https://arxiv.org/pdf/1812.06127.pdf
#     Run notebooks/create_datasets.ipynb to generate data
#     """
#     def __init__(self, dataset, client_id=None):
#         self.num_clients = 30
#         self.data = np.load(f'../data/{dataset}.npz')
#         self.x = torch.from_numpy(self.data['x']).float()
#         self.y = torch.from_numpy(self.data['y'])
#         self.n = self.data['n']
#         self.set_client(client_id)

#     def set_client(self, index=None):
#         if index is None:
#             self.client_id = None
#             self.length = len(self.y)
#         else:
#             if index < 0 or index >= self.num_clients:
#                 raise ValueError('Number of clients is out of bounds.')
#             self.client_id = index
#             self.length = self.n[index + 1] - self.n[index]

#     def __getitem__(self, index):
#         if self.client_id is None:
#             actual_index = index
#         else:
#             actual_index = self.n[self.client_id] + index
#         x, y = self.x[actual_index], self.y[actual_index]
#         return x, y

#     def __len__(self):
#         return self.length
