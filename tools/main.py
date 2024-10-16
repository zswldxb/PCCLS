import os
import sys
sys.path.insert(0, '../')
import argparse
import datetime
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
from tools.utils import set_random_seed, IOStream, load_config
import sklearn.metrics as metrics
from build import build_dataloader, build_model, build_loss, build_optimizer, build_scheduler


def train(cfg, io, exp_path):
    io.print(f"Let's use {torch.cuda.device_count()} GPUs!")
    train_loader = build_dataloader(cfg, 'train')
    test_loader = build_dataloader(cfg, 'test')
    model = nn.DataParallel(build_model(cfg.model).cuda())
    criterion = build_loss(cfg.loss).cuda()
    optimizer = build_optimizer(cfg.optimizer, model)
    scheduler = build_scheduler(cfg, optimizer)

    best_test_oa, best_test_macc = 0.0, 0.0
    for epoch in range(cfg.epochs):
        # ---------------------------------------- Train ---------------------------------------- #
        train_loss, count = 0.0, 0.0
        model.train()
        train_true_cls, train_pred_cls = [], []
        for i, (data, cls) in enumerate(train_loader):
            data, cls_true = data.cuda(non_blocking=True), cls.cuda(non_blocking=True)
            batch_size = data.size(0)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, cls_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
            optimizer.step()
            cls_pred = logits.max(dim=-1)[1]                                # (batch_size)
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true_cls.append(cls_true.cpu().numpy())                   # (batch_size)
            train_pred_cls.append(cls_pred.detach().cpu().numpy())          # (batch_size)

        scheduler.step()
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_oa = metrics.accuracy_score(train_true_cls, train_pred_cls)
        train_macc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        io.print(f'Train {epoch}: loss: {train_loss/count:.5f}, train oa: {train_oa:.5f}, train macc: {train_macc:.5f}')

        # ---------------------------------------- Test ---------------------------------------- #
        with torch.no_grad():
            test_loss, count = 0.0, 0.0
            model.eval()
            test_true_cls, test_pred_cls = [], []
            for i, (data, cls) in enumerate(test_loader):
                data, cls_true = data.cuda(non_blocking=True), cls.cuda(non_blocking=True)
                batch_size = data.size(0)
                logits = model(data)
                loss = criterion(logits, cls_true)
                cls_pred = logits.max(dim=-1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true_cls.append(cls_true.cpu().numpy())                # (batch_size)
                test_pred_cls.append(cls_pred.detach().cpu().numpy())       # (batch_size)

            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_oa = metrics.accuracy_score(test_true_cls, test_pred_cls)
            test_macc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            if test_oa >= best_test_oa:
                best_test_oa = test_oa
                torch.save(model.state_dict(), f'{exp_path}/model_oa.pth')
            if test_macc >= best_test_macc:
                best_test_macc = test_macc
                torch.save(model.state_dict(), f'{exp_path}/model_macc.pth')
            io.print(f'Test {epoch}: loss: {test_loss/count:.5f}, test oa: {test_oa:.5f}, test macc: {test_macc:.5f}, best oa: {best_test_oa:.5f}, best macc: {best_test_macc:.5f}')


@torch.no_grad()
def test(cfg, io, exp_path):
    io.print(f"Let's use {torch.cuda.device_count()} GPUs!")
    test_loader = build_dataloader(cfg, 'test')
    model.load_state_dict(torch.load(cfg.experiment.pretrain_path))
    model = nn.DataParallel(build_model(cfg.model).cuda())
    model = model.eval()

    test_true_cls, test_pred_cls = [], []
    for data, cls in test_loader:
        data, cls_true = data.cuda(non_blocking=True), cls.cuda(non_blocking=True)
        logits = model(data)
        cls_pred = logits.max(dim=-1)[1]
        test_true_cls.append(cls_true.cpu().numpy())
        test_pred_cls.append(cls_pred.detach().cpu().numpy())

    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_oa = metrics.accuracy_score(test_true_cls, test_pred_cls)
    test_macc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    io.print(f'Test last epoch: test acc: {test_oa:.5f}, test macc: {test_macc:.5f}')


if __name__ == "__main__":
    for i in range(10):
        parser = argparse.ArgumentParser(description='Point Cloud Classifier')
        parser.add_argument('--cfg', type=str, required=True, help='config file')
        args = parser.parse_args()
        cfg = EasyDict(load_config(args.cfg))
        cfg.update({'use_tqdm': False})
        assert torch.cuda.is_available() and cfg.mode in ['train', 'test']
        cfg.seed = cfg.get('seed', np.random.randint(0, 10000))
        set_random_seed(cfg.seed)
        exp_root_name, exp_base_name = ''.join(os.path.splitext(args.cfg)[0].split('cfgs/')[1:]).split('/')
        exp_path = f'{cfg.experiment.exp_root_path}/{exp_root_name}/{exp_base_name}_{cfg.seed:04d}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
        os.system(f"mkdir -p {exp_path}")
        io = IOStream(f'{exp_path}/{cfg.mode}.log')
        io.print(str(cfg))
        io.print(f'random seed is: {cfg.seed}')
        io.print(f'Using GPU: {torch.cuda.current_device()} from {torch.cuda.device_count()} devices')
        if cfg.mode == 'train':
            train(cfg, io, exp_path)
        else:
            test(cfg, io, exp_path)
            