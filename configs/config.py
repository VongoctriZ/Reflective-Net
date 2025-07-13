# config.py (refactored using @dataclass)
import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    name: str = 'Cifar10'
    num_classes: int = 10
    img_channels: int = 3
    norm_mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    norm_std: List[float] = field(default_factory=lambda: [0.2023, 0.1994, 0.2010])
    ntrain: int = 50000
    dummy: bool = False

    def __post_init__(self):
        if self.dummy:
            self.ntrain = 500
        if self.name.upper() == 'CIFAR100':
            self.name = 'Cifar100'
            self.num_classes = 100
            self.norm_mean = [0.5060725, 0.48667726, 0.4421305]
            self.norm_std = [0.2675421, 0.25593522, 0.27593908]

@dataclass
class VGGConfig:
    channels: List[int] = field(default_factory=lambda: [32, 64, 128, 128, 256, 256, 512, 512])
    is_maxpool: List[bool] = field(default_factory=lambda: [True, True, False, True, False, False, True, False])

@dataclass
class ResNetConfig:
    channels: List[int] = field(default_factory=lambda: [64, 64, 128, 128, 256, 256, 512, 512, 512])
    strides: List[int] = field(default_factory=lambda: [1, 1, 1, 2, 1, 2, 1, 2, 1])

@dataclass
class ExpConfig:
    exp_depths: List[int] = field(default_factory=lambda: [1])
    data_name: str = 'CIFAR10'
    nin: int = 1
    nSplit: int = 16
    vgg_exp_layers: List[int] = field(default_factory=lambda: [1, 3, 5])
    resnet_exp_layers: List[int] = field(default_factory=lambda: [1, 4, 6])

    # for reduce block
    expRed: List[int] = field(default_factory=lambda: [1, 2])
    # is_pool: bool = False

    exps: List[str] = field(default_factory=lambda: ['C', 'R', '1'])
    maxRan: float = 1.0
    miExp: bool = False

    def __post_init__(self):
        self.maxRan = 1 if self.data_name.upper() == 'CIFAR10' else 0.5

@dataclass
class TrainConfig:
    dummy: bool = False
    batch_size: int = 128
    epochs: int = 120
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_scheduler: str = 'step'
    milestones: List[int] = field(init=False)
    gamma: float = 0.1
    niter: int = 1000

    def __post_init__(self):
        if self.dummy:
            self.epochs = 10
        step = self.epochs // 3 + self.epochs // 10 + 2
        self.milestones = list(range(step, self.epochs, step))

@dataclass
class Config:
    dataset_name: str = 'CIFAR10'
    exp_depths: List[int] = field(default_factory=lambda: [1])
    dummy: bool = False
    dataset: DataConfig = field(init=False)
    vgg: VGGConfig = field(default_factory=VGGConfig)
    resnet: ResNetConfig = field(default_factory=ResNetConfig)
    train: TrainConfig = field(init=False)
    exp: ExpConfig = field(init=False)
    device: torch.device = field(init=False)

    def __post_init__(self):
        self.dataset = DataConfig(name=self.dataset_name, dummy=self.dummy)
        self.train = TrainConfig(dummy=self.dummy)
        self.exp = ExpConfig(exp_depths=self.exp_depths, data_name=self.dataset_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
