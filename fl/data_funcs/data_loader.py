import torchvision
from torchvision import transforms

from .fl_datasets import FEMNIST, FEMNISTClient, \
     FLCifar100, FLCifar100Client, \
     FLCifar10, FLCifar10Client, \
     ShakespeareFL, ShakespeareClient, SHAKESPEARE_EVAL_BATCH_SIZE, \
     SynthData, SynthDataFL
# from .fl_dataset import FakeFLDataset


CIFAR_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def load_data(path, dataset, load_trainset=True, download=True,
              client_id=None):
    dataset = dataset.lower()
    trainsets = None

    if (client_id is not None and client_id < 0) or dataset in [
            'emnist', 'full_shakespeare']:
        client_id = None

    if dataset.startswith("cifar"):  # CIFAR-10/100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ])

        if dataset == "cifar10":
            if load_trainset:
                trainsets = torchvision.datasets.CIFAR10(
                    root=path, train=True, download=download,
                    transform=transform_train)
            testset = torchvision.datasets.CIFAR10(
                root=path, train=False, download=download,
                transform=transform_test)
        elif dataset == "cifar10_fl":
            if load_trainset:
                fl_dataset = FLCifar10(
                    root=path, train=True, download=download,
                    transform=transform_train)
                trainsets = [FLCifar10Client(
                    fl_dataset, client_id=client_id)
                    for client_id in range(get_num_clients(dataset))]
            testset = torchvision.datasets.CIFAR10(
                root=path, train=False, download=download,
                transform=transform_test)
        elif dataset == "cifar100":
            if load_trainset:
                trainsets = torchvision.datasets.CIFAR100(
                    root=path, train=True, download=download,
                    transform=transform_train)
            testset = torchvision.datasets.CIFAR100(
                root=path, train=False, download=download,
                transform=transform_test)
        elif dataset == "cifar100_fl":
            if load_trainset:
                fl_dataset = FLCifar100(
                    path, train=True, transform=transform_train)
                trainsets = [FLCifar100Client(
                    fl_dataset, client_id=client_id)
                    for client_id in range(get_num_clients(dataset))]
            fl_dataset = FLCifar100(
                path, train=False, transform=transform_test)
            testset = FLCifar100Client(fl_dataset)
        else:
            raise NotImplementedError(f'{dataset} is not implemented.')
    elif dataset in ["femnist", 'emnist']:
        if load_trainset:
            fl_dataset = FEMNIST(path, train=True)
            trainsets = [FEMNISTClient(
                fl_dataset, client_id=client_id)
                for client_id in range(get_num_clients(dataset))]
        fl_dataset = FEMNIST(path, train=False)
        testset = FEMNISTClient(fl_dataset)
    elif dataset in ['shakespeare', 'full_shakespeare']:
        if load_trainset:
            fl_dataset = ShakespeareFL(path, train=True)
            trainsets = [ShakespeareClient(fl_dataset, client_id=client_id)
                         for client_id in range(get_num_clients(dataset))]
        fl_dataset = ShakespeareFL(path, train=False)
        testset = ShakespeareClient(fl_dataset, client_id=None)
    elif dataset.startswith("synth"):
        if load_trainset:
            fl_dataset = SynthDataFL(path, dataset)
            trainsets = [SynthData(fl_dataset, client_id=client_id)
                         for client_id in range(get_num_clients(dataset))]
        fl_dataset = SynthDataFL(path, dataset)
        testset = SynthData(fl_dataset, client_id=None)
    else:
        raise NotImplementedError(f'{dataset} is not implemented.')

    # centralized datasets
    if dataset in ['emnist', 'cifar10', 'cifar100', 'full_shakespeare']:
        trainsets, testset = [trainsets], testset
    return trainsets, testset


def get_test_batch_size(dataset, batch_size):
    dataset = dataset.lower()
    if dataset in ['shakespeare', 'full_shakespeare']:
        return SHAKESPEARE_EVAL_BATCH_SIZE
    if dataset in ['fashion-mnist', 'femnist', 'emnist']:
        return 1024
    if dataset in ['cifar10', 'cifar10_fl', 'cifar100', 'cifar100_fl']:
        return 256
    return batch_size


def get_num_classes(dataset):
    dataset = dataset.lower()
    if dataset in ['cifar10', 'cifar10_fl']:
        num_classes = 10
    elif dataset in ['cifar100', 'cifar100_fl']:
        num_classes = 100
    elif dataset in ['femnist', 'emnist']:
        num_classes = 62
    elif dataset == 'fashion-mnist':
        num_classes = 10
    elif dataset in ['shakespeare', 'full_shakespeare']:
        num_classes = 90
    elif dataset.startswith("synth"):
        num_classes = 10
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")
    return num_classes


def get_num_clients(dataset):
    dataset = dataset.lower()
    if dataset in ['emnist', 'cifar10', 'cifar100', 'full_shakespeare',
                   'full_mushrooms', 'full_ijcnn1', 'full_w8a', 'full_a9a',
                   'full_phishing']:
        num_clients = 1
    elif dataset == 'shakespeare':
        num_clients = 715
    elif dataset == 'femnist':
        num_clients = 3400
    elif dataset == 'cifar100_fl':
        num_clients = 500
    elif dataset == 'cifar10_fl':
        num_clients = 16
    elif dataset.startswith("synth"):
        num_clients = 30
    elif dataset in ['mushrooms', 'ijcnn1', 'w8a', 'a9a', 'phishing']:
        num_clients = 3
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")
    return num_clients
