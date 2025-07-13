import numpy as np
import torch
import torch.cuda.amp as tca
from gradcam import GradCAM

def get_pred(model, rawx, device):
    rawx = torch.from_numpy(rawx).to(device)
    outputs = model(rawx)
    _, predicted = torch.max(outputs.data, 1)
    return outputs, predicted

def get_gradcam(cfg, model, target_layers=None, use_dummy=True):
    if target_layers is None:
        target_layers = model.get_target_layers()

    if isinstance(target_layers, list):
        target_layers = [model.layers[i] for i in target_layers]

    gradcam = GradCAM(model=model, target_layers=target_layers, nsplit=cfg.exp.nSplit, use_dummy=use_dummy)
    return gradcam

def get_target_index(cfg, target_idx, correct_class, old_targets):
    if target_idx[0] == 'C':
        target_index = correct_class
    elif target_idx[0] == 'R':
        if cfg.exp.maxRan == 1:
            target_index = np.random.choice(np.arange(cfg.dataset.num_classes))
            while target_index in old_targets:
                target_index = np.random.choice(np.arange(cfg.dataset.num_classes))
        else:
            target_index = -np.random.choice(np.arange(int(cfg.dataset.num_classes * cfg.exp.maxRan - 1))) - 1

    elif int(target_idx[0]) > 0:
        target_index = -int(target_idx[0])
    return target_index

def batch_exp(cfg, data, norm_x, exps, grad_cam, get_exp=True, target_layers=[3]):
    mask = np.zeros((1, 1, 32, 32))
    if not get_exp:
        if cfg.exp.miExp: mask = -1
        masks = [mask] * len(exps)

    with torch.amp.autocast('cuda'):
        exp_x, raw_exp_x, aids, alogs, anlogs = [], [], [], [], []
        class_ids = np.zeros(len(exps), np.int16)
        nlogs = np.zeros(len(exps), np.float16)
        logs = np.zeros(len(exps), np.float16)
        norm_x = norm_x.unsqueeze(1)
        correct_class = data[1].cpu().numpy()

        for j in range(data[0].shape[0]):
            if get_exp:         # Method gradcam.py
                masks, raw_masks, old_targets, cmax = [], [], [], -1
                for ie, target_idx in enumerate(exps):
                    target_index = get_target_index(cfg, target_idx, correct_class[j], old_targets)
                    # Further processing with target_index and other variables
                    mask, raw_mask, class_id, log, nlog = grad_cam(norm_x[j], target_index)

                    if np.isnan(np.sum(mask)):
                        print("NaN in mask, set mask to 0")
                        mask = np.zeros_like(mask, dtype=np.float16)

                    old_targets.append(class_id)
                    masks.append(mask)
                    raw_masks.append(raw_mask)

                    class_ids[ie] = class_id
                    logs[ie] = log
                    nlogs[ie] = nlog

            sta = lambda i: np.stack([r[i] for r in raw_masks], axis=1).astype(np.float16)
            fin = np.concatenate(masks, axis=1)
            fin_raw = [sta(i) for i in range(len(target_layers))]

            exp_x.append(fin)
            raw_exp_x.append(fin_raw)

            aids.append(np.copy(class_ids))
            alogs.append(np.copy(logs))
            anlogs.append(np.copy(nlogs))

    exp_input = np.concatenate(exp_x, axis=0).astype(np.float16)

    aids = np.stack(aids, axis=0).astype(np.int16)
    alogs = np.stack(alogs, axis=0).astype(np.float16)
    anlogs = np.stack(anlogs, axis=0).astype(np.float16)

    # mid exps for one layer: batch size, class for which exp, splits, h, w
    # mid exps = list with one entry per layer
    # input exps = batch size, class for which exp, layers, h, w
    # exps = np.moveaxis(exps, 1, -1)

    def get_layer(i):
        ri = np.concatenate([r[i] for r in raw_exp_x], axis=0)
        return ri   # np.moveaxis(ri, 1, -1)

    exp_mid = [get_layer(i) for i in range(len(target_layers))]

    return exp_input, exp_mid, aids, anlogs, alogs
