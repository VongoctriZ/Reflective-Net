# example for visualizing Grad-CAM heatmaps overlay input images

from utils.visualize_overlay import *
from configs.config import Config
from classifier_handling.classifier import *
from dataset_handling.dataset_utils import *

architecture='VGG'
data_name='CIFAR10'
exp_depths=[1]
runtimes=1

cfg = Config(dataset_name=data_name, exp_depths=exp_depths, dummy=True)

# get dataset
train_dataset, val_dataset, _, norm = get_full_ds(cfg)
# train and save non-reflective model
model, lcfg, loaded = get_classifier(cfg, architecture, train_dataset, val_dataset, None, norm=norm)

# set dummy to False for get full ds
# cfg = Config(dataset_name=data_name, exp_depths=exp_depths, dummy=False)
train_dataset, val_dataset, oX, _ = get_full_ds(cfg)

# create explainations on model
train_exps, test_exps = get_exps(cfg, model, train_dataset, val_dataset, norm)
train_exps, test_exps = select_layers(train_exps), select_layers(test_exps)

# Lấy các thành phần từ test_exps sau khi select_layers
images = torch.tensor(test_exps[0])             # [N, 3, H, W]
labels = test_exps[1]                           # [N]
heatmaps = torch.tensor(test_exps[2])           # [N, H, W] — ví dụ lớp Low
class_ids = test_exps[3]                        # [N] — lớp được dùng để giải thích

# Lấy một số ảnh để visualize
indices = [31, 32, 34]

visualize_explanations(
    images[indices],
    heatmaps[indices],
    labels[indices],
    class_ids_list=class_ids[indices],
    mean=cfg.dataset.norm_mean,
    std=cfg.dataset.norm_std,
    show=True,
    save=False,
    show_original=True,
    index_names=["Correct", "Random", "Pred"],
)

