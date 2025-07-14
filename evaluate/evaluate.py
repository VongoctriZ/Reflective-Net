from configs.config import *
from dataset_handling.dataset_utils import get_full_ds, gds
from classifier_handling.classifier import *


def print_executing_config(cfg, architecture, data_name):
    print('Executing config:')
    print('CNN Architecture:', architecture.upper())
    print('Dataset:', data_name.upper())
    exp_names = ['Low', 'Middle', 'High']
    exp_layers = []
    if architecture.upper()=='VGG':
        exp_layers = cfg.exp.vgg_exp_layers
    elif architecture.upper()=='RESNET':
        exp_layers = cfg.exp.resnet_exp_layers
    exp_depths = [(exp_names[i], exp_layers[i]) for i in cfg.exp.exp_depths]
    print('Depths of explainations:', exp_depths)
    print('--------------------------------------------')


def print_executing_non_self_config(cfg, architecture_c0, architecture_ref, data_name):
    print('Executing config:')
    print('C0 Architecture:', architecture_c0.upper(), '\t Reflective Architecture:', architecture_ref.upper())
    print('Dataset:', data_name.upper())
    exp_names = ['Low', 'Middle', 'High']
    exp_layers = []
    if architecture_c0.upper()=='VGG':
        exp_layers = cfg.exp.vgg_exp_layers
    elif architecture_c0.upper()=='RESNET':
        exp_layers = cfg.exp.resnet_exp_layers
    exp_depths = [(exp_names[i], exp_layers[i]) for i in cfg.exp.exp_depths]
    print('Depths of explainations:', exp_depths)
    print('--------------------------------------------')

def evaluate_self_run(cfg, architecture='VGG', all_gains=[]):
    # all_gains = []

    # get dataset
    train_dataset, val_dataset, _, norm = get_full_ds(cfg)

    # train and save non-reflective model
    model, lcfg, loaded = get_classifier(cfg, architecture, train_dataset, val_dataset, None, norm=norm)

    # create explainations on model
    train_dataset, val_dataset, oX, _ = get_full_ds(cfg)
    train_exps, test_exps = get_exps(cfg, model, train_dataset, val_dataset, norm)
    train_exps, test_exps = select_layers(train_exps), select_layers(test_exps)

    # Train new reflective model
    train_dataset, val_dataset = gds(train_exps, cfg.train), gds(test_exps, cfg.train)
    exp_model, ecfg = get_exp_classifier(cfg, architecture, train_dataset, val_dataset, None, trained_net_self=model)

    # return exp_model, all_gains, ecfg, lcfg

    print('Outcomes:')
    print('Accuracies on non-reflective Classifier:', lcfg)
    print('Accuracies on reflective Classifier:', ecfg)

    all_gains.append(ecfg['test_acc_predict'] - lcfg['test_acc'])

    print('Gains of reflective Classifier if use predictions in [%] for this run:', np.round(all_gains[-1], 4)*100)
    print('... for all runs', '#runs:', len(all_gains),
          'Mean Gain', np.round(np.mean(np.array(all_gains)), 4)*100,
          'Std Gain', np.round(np.std(np.array(all_gains)), 4)*100)
    print('\n\n')

def evaluate_self(architecture='VGG', data_name='CIFAR10', exp_depths=[1], runtimes=1):
    all_gains = []
    cfg = Config(dataset_name=data_name, exp_depths=exp_depths, dummy=True)

    print_executing_config(cfg, architecture, data_name)

    for i in range(runtimes):
        evaluate_self_run(cfg, architecture, all_gains)


def evaluate_non_self_run(cfg, architecture_c0='VGG', architecture_ref='RESNET',  all_gains=[]):
    # all_gains = []

    # get dataset
    train_dataset, val_dataset, _, norm = get_full_ds(cfg)

    # train and save non-reflective model
    model, lcfg, loaded = get_classifier(cfg, architecture_c0, train_dataset, val_dataset, None, norm=norm)

    # create explainations on model
    train_dataset, val_dataset, oX, _ = get_full_ds(cfg)
    train_exps, test_exps = get_exps(cfg, model, train_dataset, val_dataset, norm)
    train_exps, test_exps = select_layers(train_exps), select_layers(test_exps)

    # Train new reflective model
    train_dataset, val_dataset = gds(train_exps, cfg.train), gds(test_exps, cfg.train)
    exp_model, ecfg = get_exp_classifier(cfg, architecture_ref, train_dataset, val_dataset, None, trained_net_self=None)

    # return exp_model, all_gains, ecfg, lcfg

    print('Outcomes:')
    print('Accuracies on non-reflective Classifier:', lcfg)
    print('Accuracies on reflective Classifier:', ecfg)

    all_gains.append(ecfg['test_acc_predict'] - lcfg['test_acc'])

    print('Gains of reflective Classifier if use predictions in [%] for this run:', np.round(all_gains[-1], 4)*100)
    print('... for all runs', '#runs:', len(all_gains),
          'Mean Gain', np.round(np.mean(np.array(all_gains)), 4)*100,
          'Std Gain', np.round(np.std(np.array(all_gains)), 4)*100)
    print('\n\n')

def evaluate_non_self(architecture_c0='VGG', architecture_ref='RESNET', data_name='CIFAR10', exp_depths=[1], runtimes=1):
    all_gains = []
    cfg = Config(dataset_name=data_name, exp_depths=exp_depths, dummy=True)

    print_executing_non_self_config(cfg, architecture_c0, architecture_ref, data_name)

    for i in range(runtimes):
        evaluate_non_self_run(cfg, architecture_c0, architecture_ref, all_gains)

if __name__ == '__main__':
    evaluate_self(architecture='vgg', data_name='cifar10', exp_depths=[1], runtimes=1)
    