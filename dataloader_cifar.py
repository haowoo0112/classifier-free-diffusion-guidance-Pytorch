from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler

def load_data(batchsize:int, numworkers:int):
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = CIFAR10(
                        root = './',
                        train = True,
                        download = False,
                        transform = trans
                    )
    trainloader = DataLoader(
                        data_train,
                        batch_size = batchsize,
                        num_workers = numworkers,
                        drop_last = True
                    )
    return trainloader

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5

