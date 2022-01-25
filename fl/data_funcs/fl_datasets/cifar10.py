from PIL import Image
import os
import pickle
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class FLCifar10Client(Dataset):
    def __init__(self, fl_dataset, client_id=None):

        self.fl_dataset = fl_dataset
        self.set_client(client_id)

    def set_client(self, index=None):
        fl = self.fl_dataset
        if index is None:
            self.client_id = None
            self.length = len(fl.data)
            self.data = fl.data
        else:
            if index < 0 or index >= fl.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            indices = fl.partition[self.client_id]
            self.length = len(indices)
            self.data = fl.data[indices]
            self.targets = [fl.targets[i] for i in indices]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        fl = self.fl_dataset
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other fl_datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if fl.transform is not None:
            img = fl.transform(img)

        if fl.target_transform is not None:
            target = fl.target_transform(target)

        return img, target

    def __len__(self):
        return self.length


class FLCifar10(CIFAR10):
    """
    CIFAR10 Dataset.
    100 clients that were allocated data_preprocess uniformly at random.
    Run notebooks/create_datasets.ipynb to generate data split
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(FLCifar10, self).__init__(root, train=train, transform=transform,
                                        target_transform=target_transform,
                                        download=download)

        partition_dir = os.path.join(root, "cifar10.pkl")
        self.partition = pickle.load(open(partition_dir, "rb"))
        self.num_clients = len(self.partition)


# class FLCifar10(CIFAR10):
#     """
#     CIFAR10 Dataset.
#     100 clients that were allocated data_preprocess uniformly at random.
#     Run notebooks/create_datasets.ipynb to generate data split
#     """
#     def __init__(
#       self, root, train=True, transform=None, target_transform=None,
#       download=False, client_id=None):

#         super(FLCifar10, self).__init__(
#           root, train=train, transform=transform,
#           target_transform=target_transform,
#           ownload=download)
#         self.num_clients = 100
#         self.dataset_indices = np.arange(len(self.data))
#         if train:
#             self.dataset_indices = np.load('../data/cifar10_ids.npy')
#         self.n_client_samples = len(self.data) // self.num_clients
#         self.set_client(client_id)

#     def set_client(self, index=None):
#         if index is None:
#             self.client_id = None
#             self.length = len(self.data)
#         else:
#             if index < 0 or index >= self.num_clients:
#                 raise ValueError('Number of clients is out of bounds.')
#             self.client_id = index
#             self.length = self.n_client_samples

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         if self.client_id is None:
#             actual_index = index
#         else:
#             client_id = int(self.client_id)
#             actual_index = client_id * self.n_client_samples + index
#         img, target = self.data[actual_index], self.targets[actual_index]

#         # doing this so that it is consistent with all other fl_datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return self.length
