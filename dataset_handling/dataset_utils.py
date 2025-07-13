import torchvision,torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset
import numpy as np
import sklearn
from configs.config import Config

# get dataset
gds = lambda dataset, cfg: torch.utils.data.DataLoader(TensorDataset(*[torch.from_numpy(x) for x in dataset]), batch_size=cfg.batch_size)

def get_norm(mean, std):
    return (torch.from_numpy(np.array(mean, np.float32).reshape(1,3,1,1)).cuda(),
            torch.from_numpy(np.array(std, np.float32).reshape(1,3,1,1)).cuda())

def get_full_ds(cfg):
    if cfg.dataset.name.upper() == 'CIFAR10':
        data = torchvision.datasets.CIFAR10
    elif cfg.dataset.name.upper() == 'CIFAR100':
        data = torchvision.datasets.CIFAR100

    trans=transforms.Compose([transforms.ToTensor()])
    ntrain, down = cfg.dataset.ntrain, True

    def load_store(is_train, n_data):
        nonlocal data
        trainset = data(root="data/", train=is_train, download=down, transform=transforms.ToTensor())
        train_dataset = torch.utils.data.DataLoader(trainset, batch_size=n_data, num_workers=4)

        ds = next(iter(train_dataset))
        X, Y = ds[0].clone().numpy(), ds[1].clone().numpy()
        ds = [X, Y]
        ds = sklearn.utils.shuffle(*ds)
        return ds[0].astype(np.float16), ds[1].astype(np.int16)

    train_X, train_Y = load_store(True, ntrain)
    test_X, test_Y = load_store(False, ntrain//2)

    def create_dataset(cfg, train_X, train_Y, shuffle=True):
        ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_Y))
        return torch.utils.data.DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle, num_workers=4)

    trainset = create_dataset(cfg, train_X, train_Y)
    testset = create_dataset(cfg, test_X, test_Y, False)

    norm = get_norm(cfg.dataset.norm_mean, cfg.dataset.norm_std)

    return trainset, testset, None, norm
