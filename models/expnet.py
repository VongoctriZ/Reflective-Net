import numpy as np
import torch
import torch.nn as nn
from configs.config import Config
from models.blocks import *
from typing import Optional

class ExpNet(nn.Module):
    def __init__(self, cfg: Config, architecture: str = 'vgg', isExp: bool = False, num_classes: Optional[int] = None):
        """ Initializes the ExpNet model.
        Args:
            cfg (Config): Configuration object containing model parameters.
            architecture (str): Architecture type, either 'vgg' or 'resnet'.
            isExp (bool): Flag indicating if the model is an explanation model.
            num_classes (Optional[int]): Number of output classes. If None, uses cfg.dataset.num_classes.
        """
        super().__init__()

        self.cfg = cfg
        self.architecture = architecture.lower()
        self.isExp = isExp
        self.num_classes = num_classes or cfg.dataset.num_classes
        self.in_channels = 3

        if self.architecture == 'vgg':
            self.channels = cfg.vgg.channels
            self.is_maxpool = cfg.vgg.is_maxpool
            self.targets = cfg.exp.vgg_exp_layers + [9999]
        elif self.architecture == 'resnet':
            self.channels = cfg.resnet.channels
            self.strides = cfg.resnet.strides
            self.targets = cfg.exp.resnet_exp_layers + [9999]
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        self.base_channels = [self.in_channels] + self.channels
        self.offsets = np.zeros(len(self.base_channels), dtype=int)  # Cộng thêm channel nếu có reduce

        # thiết lập các target GradCAM layer và số feature cần thêm
        self.nExtra = 0

        if self.isExp: self._init_gradcam_config()

        self._build_backbone()

        if self.isExp: self._build_reduce_blocks()

        # Xác định số kênh cuối cùng trước classifier
        final_channels = self.base_channels[-1] + self.offsets[-1]
        self.classifier = ClassifierBlock(final_channels, self.num_classes)

    def _init_gradcam_config(self):
        """Khởi tạo thông tin về nhánh giải thích, số features thêm, và target layers"""
        assert self.num_classes is not None
        self.oney = torch.eye(self.num_classes).cuda()

        self.nin = self.cfg.exp.nin
        self.nSplit = self.cfg.exp.nSplit
        self.nExtra = self.nin * self.nSplit

        # Lấy danh sách target layers từ config
        if self.architecture == 'vgg':
            layers = np.array(self.cfg.exp.vgg_exp_layers) + 1     # +1 vì layer đầu tiên thường là input
        elif self.architecture == 'resnet':
            layers = np.array(self.cfg.exp.resnet_exp_layers) + 1  # +1 vì layer đầu tiên thường là input
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        # Chọn depth (subset các layer)
        target_layers = layers[self.cfg.exp.exp_depths]
        self.targets = target_layers.tolist() + [9999]  # sentinel

        # Gán offsets cho từng layer được chọn để biết cần thêm bao nhiêu channel
        red_factor = np.prod(self.cfg.exp.expRed) if self.cfg.exp.expRed else 1
        added = max(1, self.nExtra // red_factor)
        for idx in target_layers:
            self.offsets[idx] = added  
    
    def _build_backbone(self):
        if self.architecture == 'vgg':
            self.features = self._build_vgg()
        elif self.architecture == 'resnet':
            self.features = self._build_resnet()

    def _build_vgg(self):
        self.layers = nn.ModuleList()
        for i in range(len(self.base_channels) - 1):
            in_c = self.base_channels[i] + self.offsets[i] if i > 0 else self.base_channels[i]
            out_c = self.base_channels[i + 1]

            block = VGG_ConvBlock(in_c, out_c, kernel_size=3, stride=1,
                                  maxpool=self.is_maxpool[i])
            self.layers.append(block)

    def _build_resnet(self):
        self.layers = nn.ModuleList()
        # First conv layer
        stem_block = VGG_ConvBlock(self.base_channels[0], self.base_channels[1], kernel_size=3, stride=1, maxpool=False)
        self.layers.append(stem_block)

        for i in range(1, len(self.base_channels) - 2, 2):
            in_c = self.base_channels[i] + self.offsets[i] if i > 0 else self.base_channels[i]
            mid_c = self.base_channels[i + 1]
            out_c = self.base_channels[i + 2]
            block = ResNet_ConvBlock(in_c, mid_c, out_c, stride=self.strides[i], offset=self.offsets[i+1])
            self.layers.append(block.conv1)
            self.layers.append(block.conv2)

    def _build_reduce_blocks(self):
        self.reduce_block = ReduceBlock(in_channels=self.nExtra,
                                        red1=self.cfg.exp.expRed[0],
                                        red2=self.cfg.exp.expRed[1],
                                        architecture=self.architecture
                                       )

    def rsh(self, cx):
        s = cx.shape
        return torch.reshape(cx, (s[0], s[1] * s[2], s[3], s[4]))

    def preprocess(self, xbatch):
        # Normalize the input tensor
        xExp, aExp = None, {}
        if self.isExp:
            x = xbatch[0]
            nexp = np.sum(np.array(self.targets[:-1]) > 0)       # num exp layer
            xExp = [ self.rsh(cx) for cx in xbatch[1][:nexp] ]   # use for gradcam
        else:
            x = xbatch
            aExp = None
        return x, xExp, aExp

    def handle_exp_layer(self, weights, x, xbatch, aExp):
        aweis = []
        for j in range(self.nin):
            cur_aweis = [ weights[:, j * self.nSplit : (j+1) * self.nSplit] ]
            cur_aweis = torch.cat(cur_aweis, dim=1)
            aweis.append(cur_aweis)

        weights = torch.cat(aweis, dim=1)
        weights = self.reduce_block(weights)

        x = torch.cat([x, weights], axis=1)
        return x

    def forward(self, xbatch):
        x, xExp, aExp = self.preprocess(xbatch)
        tpos = 0

        for i, layer in enumerate(self.layers):
            if self.isExp and i == self.targets[tpos]:
                weights = xExp[tpos]
                x = self.handle_exp_layer(weights, x, xbatch, aExp)
                tpos += 1
            x = layer(x)
        x = self.classifier(x)
        return x

    def get_target_layers(self):
        target_layers = [self.layers[i] for i in self.targets[:-1]]
        return target_layers