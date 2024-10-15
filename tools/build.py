import torch
import torch.nn as nn
import pccls.datasets as datasets
import pccls.models as models
import pccls.losses as losses
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import copy


# ---------------------------------------- Build Dataset ---------------------------------------- #
def build_dataset(cfg_dataset):
    cfg_dataset = copy.deepcopy(cfg_dataset)
    NAME = cfg_dataset.pop('NAME')
    return datasets.__all__[NAME](**cfg_dataset)

# ---------------------------------------- Build Dataloader ---------------------------------------- #
def build_dataloader(cfg, split='train'):
    assert split in ['train', 'test']
    cfg = copy.deepcopy(cfg)
    cfg_dataset, cfg_dataset_split, cfg_dataloader = cfg.dataset.common, cfg.dataset.get(split), cfg.dataloader.get(split)
    cfg_dataset.update(cfg_dataset_split)
    dataset = build_dataset(cfg_dataset)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=(split=='train'), drop_last=(split=='train'), num_workers=8, **cfg_dataloader)
    return dataloader

# ---------------------------------------- Build Model ---------------------------------------- #
def build_model(cfg_model):
    cfg_model = copy.deepcopy(cfg_model)
    NAME = cfg_model.pop('NAME')
    return models.__dict__[NAME](cfg_model)

# ---------------------------------------- Build Loss ---------------------------------------- #
def build_loss(cfg_loss):
    cfg_loss = copy.deepcopy(cfg_loss)
    NAME = cfg_loss.pop('NAME')
    return losses.__dict__[NAME](**cfg_loss)

# ---------------------------------------- Build Optimizer ---------------------------------------- #
def build_optimizer(cfg_optimizer, model):
    assert isinstance(model, nn.Module) and cfg_optimizer.NAME in ['SGD', 'Adam'] and 'lr' in cfg_optimizer.keys()
    cfg_optimizer = copy.deepcopy(cfg_optimizer)
    NAME = cfg_optimizer.pop('NAME')
    if NAME == 'SGD':
        optimizer = optim.SGD(model.parameters(), **cfg_optimizer)
    elif NAME == 'Adam':
        optimizer = optim.Adam(model.parameters(), **cfg_optimizer)
    return optimizer

# ---------------------------------------- Build Scheduler ---------------------------------------- #
def build_scheduler(cfg, optimizer):
    cfg_scheduler, epoch = copy.deepcopy(cfg.scheduler), cfg.epochs
    NAME = cfg_scheduler.pop('NAME')
    assert isinstance(optimizer, optim.Optimizer) and NAME in ['cos', 'step']
    if NAME == 'cos':
        scheduler = sched.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, **cfg_scheduler)
    elif NAME == 'step':
        scheduler = sched.MultiStepLR(optimizer=optimizer, **cfg_scheduler)
    return scheduler
