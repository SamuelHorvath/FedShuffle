# from torch.utils.data import Dataset


# class FLDataset(Dataset):
#     """
#     Base class for Federated Datasets
#     with pointers to clients.
#     """
#     def set_client(self, index=None):
#         raise NotImplementedError


# class FakeFLDataset:
#     def __init__(self, centralized_dataset):
#         self.dataset = centralized_dataset
#         self.num_clients = 1

#     def __len__(self):
#         return len(self.dataset)

#     def set_client(self, index=None):
#         pass

#     def __getitem__(self, index):
#         return self.dataset.__getitem__(index)
