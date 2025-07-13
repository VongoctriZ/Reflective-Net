import numpy as np,copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as tca

from models.expnet import ExpNet
from explain.gradcam_utils import *

# experiment run
def exp_run(cfg, dataset, get_exp, nexp, exps, grad_cam, target_layers, norm):
    # Data consists of: X, y, all exps, estimated y of exps, logit y, normed logits
    # All Exps: List of Input Ex, Mid Ex (where Input Ex is always one entry, and Mid Ex has one list entry per layer)
    # Mid Ex for one layer: batchSize, ClassForWhichExp, splits, h, w
    # Input Ex: batchSize, ClassForWhichExp, layers, h, w

    ox, oy, exp_x, raw_exp_x, masks = [], [], [], [], []
    aids, alogs, anlogs = [], [], []

    for i, data in enumerate(dataset):
        norm_x = (data[0].cuda() - norm[0]) / norm[1]
        ox.append(norm_x.cpu().numpy().astype(np.float16))
        oy.append(data[1].numpy().astype(np.int16))

        bx, bx2, clids, logs, nlogs = batch_exp(cfg, data, norm_x, exps, grad_cam, get_exp=get_exp, target_layers=target_layers)
        # print(f'batch{i}.shape', bx.shape)
        exp_x.append(bx)
        raw_exp_x.append(bx2)

        aids.append(clids)
        alogs.append(logs)
        anlogs.append(nlogs)

        if len(oy) * data[0].shape[0] > nexp: break
        if len(oy) % 40 == 0: print("Computed Explanations: ", len(oy) * data[0].shape[0])

    oy = np.concatenate(oy).astype(np.int16)
    ox = np.concatenate(ox, axis=0).astype(np.float16)
    exp_x = np.concatenate(exp_x, axis=0).astype(np.float16)

    aids = np.concatenate(aids, axis=0).astype(np.int16)
    alogs = np.concatenate(alogs, axis=0).astype(np.float16)
    anlogs = np.concatenate(anlogs, axis=0).astype(np.float16)

    sta = lambda i: np.concatenate([r[i] for r in raw_exp_x], axis=0)
    exp_r = [sta(i) for i in range(len(target_layers))]
    return ox, oy, [exp_x] + exp_r, aids, alogs, anlogs

def select_layers(ds):        # Returns X, Y, Exp(SalMaps), classes, logits (if used later)
    tl = [0]
    ds = list(ds[:2]) + ds[2] + list(ds[3:])
    return ds[:2] + [ds[3 + t] for t in tl] + [ds[-3]]

# -------------------------------------- Get Explanations ------------------------------------
def get_exps(cfg, model, train_dataset, val_dataset, norm):
    gradcam = None

    target_layers = np.array(model.targets)[cfg.exp.exp_depths]
    target_layers = target_layers.tolist()
    print('target_layer for gradcam:', target_layers)

    gradcam = get_gradcam(cfg, model, target_layers)

    print("Compute Explanations for training data")
    train_data = exp_run(cfg, train_dataset, True, cfg.dataset.ntrain, cfg.exp.exps, gradcam, target_layers, norm)

    print("Compute Explanations for validation data")
    val_data = exp_run(cfg, val_dataset, True, cfg.dataset.ntrain // 2, cfg.exp.exps, gradcam, target_layers, norm)

    return train_data, val_data

def get_exp(data, ran_opt, zero_exp, is_train=True):
    idxs =  None
    if is_train:
        idxs = np.random.choice(ran_opt, data[0].shape[0])
        lex = [np.expand_dims(d[np.arange(d.shape[0]), idxs], axis=1) for d in data]
    else:
        lex = [np.expand_dims(d[:, ran_opt[0]], axis=1) for d in data]

    return lex, idxs

def get_x_data(x_data, zero_exp, aug, ran_opt):
    rdata = [x.clone().numpy() for x in x_data[1:]] # Selected explanations - don't change original input at 0
    ex, idxs = get_exp(rdata, ran_opt, zero_exp=zero_exp)
    return [x_data[0].cuda()]+[torch.from_numpy(d).cuda() for d in ex], idxs

# -------------------------- Get Accuracy - Loss ----------------------------------

def get_sigle_acc(net, data_x, labels, pool=None):
    with torch.amp.autocast('cuda'):
        outputs = net(data_x)
        if type(outputs) is tuple: outputs=outputs[1] #for attention net use second output
        _, predicted = torch.max(outputs.data, 1)
        correct = torch.eq(predicted, labels).sum().item()
        return correct

def get_exp_acc(net, dataset, iexp, niter=100, pool=None, zero_exp=1):
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset):
            labels = data[1].cuda()
            exp_data = [d.clone().numpy() for d in data[2:]] # Selected explanations - don't change original input at 0
            nd, _ = get_exp(exp_data, iexp, zero_exp=zero_exp, is_train=False)

            xgpu = data[0].cuda()
            ndgpu = [torch.from_numpy(x).cuda() for x in nd]

            correct += get_sigle_acc(net, (xgpu, ndgpu), labels, pool=pool)
            total += labels.size(0)
            if i >= niter: break
    return correct / total

def get_acc(net, dataset, niter=100, norm=None):
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset):
            dsx, dsy = data[0].cuda(), data[1].cuda()
            dsx = (dsx - norm[0]) / norm[1]
            total += dsy.size(0)
            outputs = net(dsx.float())
            _, predicted = torch.max(outputs.data, 1)
            correct += torch.eq(predicted, dsy).sum().item()
            if i >= niter: break
    return correct / total

def get_loss(model):
    reg_loss = 0
    for name,param in model.named_parameters():
        if 'bn' not in name:
             reg_loss += torch.norm(param)
    # loss = cls_loss + args.weight_decay*reg_loss
    return reg_loss

# -------------------------------------- Get Classifier ------------------------------------

def get_out(ndgpu, exp_net):
    dropCl = False
    dropA =  False
    output = exp_net((ndgpu[0], ndgpu[1:], dropCl, dropA))
    return output

def get_net(cfg, architecture, isExp=False):
    NETWORK = ExpNet
    exp_net = NETWORK(cfg, architecture, isExp, cfg.dataset.num_classes).cuda()
    return exp_net

def get_exp_classifier(cfg, architecture, train_dataset, val_dataset, res_folder, trained_net_self=None):
    exp_net = get_net(cfg, architecture, isExp=True)

    if trained_net_self is not None: # copying params     
        exp_params = dict(exp_net.named_parameters())
        trained_params = dict(trained_net_self.named_parameters())

        # print_info('trained net (Self)', trained_net_self)
        # print_info('exp net (Reflective)', exp_net)

        for name, trained_param in trained_params.items():
            if name.startswith('layers') or name.startswith('classifier.'):
                if name in exp_params:
                    exp_param = exp_params[name]

                    if len(trained_param.shape) != len(exp_param.shape):
                        print(f"Warning: Shape mismatch for parameter {name}: trained {trained_param.shape} vs exp {exp_param.shape}")
                        continue

                    if trained_param.data.shape == exp_param.data.shape:
                        # Shapes match, copy all data
                        exp_param.data.copy_(trained_param.data)

                    elif len(trained_param.shape) == 4: # Conv2d weights (out_channels, in_channels, kernel_h, kernel_w)
                        # Copy weights for original input channels (dimension 1)
                        in_c_orig = trained_param.data.shape[1]
                        exp_param.data[:, :in_c_orig, :, :] = trained_param.data.clone()

                    elif len(trained_param.shape) == 2: # Linear weights (out_features, in_features)
                        # Copy weights for original input features (dimension 1)
                        in_f_orig = trained_param.data.shape[1]
                        exp_param.data[:, :in_f_orig] = trained_param.data.clone()
                    else:
                        # Handle other dimensions if necessary (e.g., 1D bias)
                        # If shapes are different but dimensions match, assume it's a simple bias mismatch
                        # Copy based on minimum shape if dimensions allow
                        min_shape = [min(s1, s2) for s1, s2 in zip(trained_param.data.shape, exp_param.data.shape)]
                        # Construct slice dynamically
                        slices = [slice(0, ms) for ms in min_shape]
                        exp_param.data[slices] = trained_param.data[slices].clone()

    exp_optimizer = optim.SGD(exp_net.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)

    exp_scheduler = optim.lr_scheduler.MultiStepLR(
        exp_optimizer,
        milestones=cfg.train.milestones,
        gamma=cfg.train.gamma
    )

    closs, epochs, loss = 0, cfg.train.epochs, nn.CrossEntropyLoss()

    print("Train Reflective Classifier")
    scaler = torch.amp.GradScaler('cuda')
    ran_opt = np.sort(np.array([0, 1])) #sorting is very important
    emate_accs, etr_acc = [], []

    iexp = list(np.arange(len(cfg.exp.exps)))
    icorr = iexp[:1] + iexp[2:]
    imax = iexp[2:]

    for epoch in range(epochs):
        exp_net.train()
        for i, data in enumerate(train_dataset):
            with torch.amp.autocast('cuda'):
                exp_optimizer.zero_grad()
                dsy = data[1].cuda()
                ndgpu, idxs = get_x_data([data[0]] + list(data[2:]), False, False, ran_opt)
                output = get_out(ndgpu, exp_net)
                errD_real = loss(output, dsy.long())

                scaler.scale(errD_real).backward()
                scaler.step(exp_optimizer)
                scaler.update()

                closs = 0.97 * closs + 0.03 * errD_real.item() if epoch > 20 else 0.8 * closs + 0.2 * errD_real.item()

        # update lr if needed
        exp_scheduler.step()

        exp_net.eval()

        emate_accs.append(get_exp_acc(exp_net, val_dataset, imax, niter=cfg.train.niter) if len(imax) else -1)

        if (epoch % 4 == 0 and epoch <= 13) or (epoch % 20 == 0 and epoch > 13) :
            cur_acc = get_exp_acc(exp_net, val_dataset, iexp[1:], niter=cfg.train.niter)
            print(epoch, np.round(np.array([closs, cur_acc, get_exp_acc(exp_net, train_dataset, icorr, niter=cfg.train.niter)]), 5))

            if np.isnan(closs):
                print("Failed!!!")
                return None, None

    exp_net.eval()
    lcfg = {
        'test_acc_correct': get_exp_acc(exp_net, val_dataset, [0], niter=cfg.train.niter),
        'test_acc_predict': get_exp_acc(exp_net, val_dataset, [2], niter=cfg.train.niter),
        'test_acc_random' : get_exp_acc(exp_net, val_dataset, [1], niter=cfg.train.niter),
    }

    set_eval(exp_net)

    return exp_net, lcfg

def set_eval(exp_net):
    exp_net.eval()
    for name, module in exp_net.named_modules():
        if isinstance(module, nn.Dropout): module.p = 0
        elif isinstance(module, nn.LSTM): module.dropout = 0 #print("zero lstm drop") #print("zero drop")
        elif isinstance(module, nn.GRU): module.dropout = 0

def get_classifier(cfg, architecture, train_dataset, val_dataset, res_folder, forceLoad=False, norm=None):
    non_exp_net = get_net(cfg, architecture, isExp=False)

    non_exp_optimizer = optim.SGD(non_exp_net.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
    non_exp_scheduler = optim.lr_scheduler.MultiStepLR(
        non_exp_optimizer,
        milestones=cfg.train.milestones,
        gamma=cfg.train.gamma
    )

    closs, epochs, loss, = 0, cfg.train.epochs, nn.CrossEntropyLoss()

    print("Train non-reflective classifier")
    scaler = torch.amp.GradScaler('cuda')
    test_accs, train_accs = [], []
    class_acc = lambda dataset:get_acc(non_exp_net, dataset, niter=cfg.train.niter, norm=norm)

    for epoch in range(epochs):
        non_exp_net.train()
        for i, data in enumerate(train_dataset):
            with torch.amp.autocast('cuda'):
                non_exp_optimizer.zero_grad()

                dsx = data[0].cuda()
                dsx = (dsx - norm[0]) / norm[1]
                dsy = data[1].cuda()

                output = non_exp_net(dsx.float())
                errD_real = loss(output, dsy.long())

                scaler.scale(errD_real).backward()
                scaler.step(non_exp_optimizer)
                scaler.update()

                closs = 0.97 * closs + 0.03 * errD_real.item() if epoch > 20 else 0.8 * closs + 0.2 * errD_real.item()

        # update lr if needed
        non_exp_scheduler.step()
        non_exp_net.eval()

        test_accs.append(class_acc(val_dataset))
        train_accs.append(class_acc(train_dataset))

        if (epoch % 4 == 0 and epoch <= 13) or (epoch % 20 == 0 and epoch > 13):
            print(epoch, np.round(np.array([closs, test_accs[-1], train_accs[-1]]), 5))

            if np.isnan(closs):
                print("Failed!!!")
                return None, None
        lcfg = {
            'test_acc': class_acc(val_dataset),
            'train_acc': class_acc(train_dataset),
        }

        set_eval(non_exp_net)

    return non_exp_net, lcfg, False
