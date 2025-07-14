import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM(nn.Module):
    def __init__(self, model, target_layers, gonly=0, nsplit=1, salnorm=True, relu=True, use_dummy=True):
        super(GradCAM, self).__init__()
        self.model = model
        self.gradients = [None for _ in range(len(target_layers))]
        self.activations = [None for _ in range(len(target_layers))]
        self.gonly, self.nsplit, self.salnorm, self.relu, self.use_dummy = gonly, nsplit, salnorm, relu, use_dummy
        # self.use_dummy =False #for debug visualize

        def addHook(target_layer, i):
            def backward_hook(module, grad_input, grad_output):
                self.gradients[i] = grad_output[0]
                return None
            def forward_hook(module, input, output):
                self.activations[i] = output
                return None
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)

        for i, target_layer in enumerate(target_layers):
            addHook(target_layer, i)

    def get_weights(self, idx, score):
        gradients = self.gradients[idx]
        b, c, h, w = gradients.size()
        alpha = gradients.view(b, c, -1).mean(2)  # GAP
        weights = alpha.view(b, c, 1, 1)
        return weights

    def get_maps(self, idx, h, w, score):
        gradients, activations = self.gradients[idx], self.activations[idx]
        weights = self.get_weights(idx, score)
        relu_func = F.relu if self.relu else lambda x: x

        saliency_map = (weights * activations)
        osaliency_map = saliency_map.detach()

        # Reduce channel dimension
        vecs = torch.split(osaliency_map, osaliency_map.shape[1] // self.nsplit, dim=1)
        vecs = [v.sum(1, keepdim=True) for v in vecs]
        csaliency_map = torch.cat(vecs, dim=1)

        # Normalize
        def norm(cmap):
            min_val, max_val = cmap.min(), cmap.max()
            return (cmap - min_val).div(max_val - min_val + 1e-8).data

        nonUpSalmap = relu_func(csaliency_map)
        nonUpSalmap = norm(nonUpSalmap)
        nonUpSalmap = nonUpSalmap.cpu().numpy().astype(np.float16)

        if self.use_dummy:
            dummy_map = np.zeros([1, 1, 32, 32], dtype=np.float16)
            return dummy_map, nonUpSalmap
        else:
            csaliency_map = csaliency_map.cpu().numpy().astype(np.float16)
            return csaliency_map, nonUpSalmap

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
            If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size() if not isinstance(input, tuple) else input[0].size()
        logit = self.model(input)

        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
            clid = logit.max(1)[-1].detach().cpu().numpy()[0]
        else:
            if class_idx >= 0:
                score = logit[:, class_idx].squeeze()
                clid = class_idx
            else:
                sorted_score = logit.sort(1).indices
                score = logit[:, sorted_score[-1][class_idx]].squeeze()
                clid = sorted_score[-1][class_idx].detach().cpu().numpy()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)

        maps, rmaps = [], []
        for i in range(len(self.gradients)):
            m, rm = self.get_maps(i, h, w, score)
            maps.append(m)
            rmaps.append(rm)

        self.model.zero_grad()
        log = logit[:, clid].detach()
        nlog = log / (1e-7 + torch.sum(torch.abs(logit)))

        return (
            np.stack(maps, axis=2),
            rmaps,
            clid,
            log.cpu().numpy()[0],
            nlog.detach().cpu().numpy()[0]
        )

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
    