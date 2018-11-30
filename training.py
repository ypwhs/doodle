from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import horovod.torch as hvd

import os
from time import time
import numpy as np


def mapk(output, target, k=3):
    """
    Computes the mean average precision at k.

    Parameters:
        :type output: torch.Tensor
        :type target: torch.int
        :type k: int

        :param output: A Tensor of predicted elements. Shape: (N,C)  where C = number of classes, N = batch size
        :param target: A Tensor of elements that are to be predicted. Shape: (N) where each value is  0≤targets[i]≤C−1
        :param k: The maximum number of predicted elements

    Returns:
        torch.float:  The mean average precision at k over the output
    """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i in range(k):
            correct[i] = correct[i] * (k - i)

        score = correct[:k].view(-1).float().sum(0, keepdim=True)
        score.mul_(1.0 / (k * batch_size))
        return score


def save_checkpoint(model, optimizer, test_acc, tag, path='checkpoints'):
    os.makedirs(path, exist_ok=True)
    filename = f'{path}/{model.name}_{test_acc:.4f}_{tag}.pt'
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)
    print('\nSave checkpoint', filename)
    return filename


def load_checkpoint(model, optimizer, path):
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])


def change_lr(optimizer, lr, weight_decay=1e-5):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay


def test(model, test_loader):
    model.eval()
    total_loss = 0.0
    total_map = 0.0
    mean_loss = 0.0
    mean_map = 0.0
    start = time()

    with torch.no_grad():
        if hvd.rank() == 0:
            pbar = tqdm(test_loader)
        else:
            pbar = test_loader

        for batch_idx, (data, target) in enumerate(pbar):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = F.cross_entropy(output, target)

            total_loss += loss
            total_loss = hvd.allreduce(total_loss)

            score = mapk(output, target)[0]

            total_map += score
            total_map = hvd.allreduce(total_map)

            mean_loss = total_loss.item() / (batch_idx + 1)
            mean_map = total_map.item() / (batch_idx + 1)

            if hvd.rank() == 0:
                pbar.set_description(f'Test Loss: {mean_loss:.4f} MAP@3: {mean_map*100.:.2f}% ')
    end = time()

    if hvd.rank() == 0:
        pbar.close()
    t = end - start
    t = f'{int(t / 60):02}:{int(t % 60):02}'
    return mean_loss, mean_map, t


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train(model, train_loader, optimizer, epoch, scheduler=None, mixup=False, mixup_alpha=0.2, class_num=340):
    mean_loss = 0
    mean_map = 0

    checkpoint_path = ''
    model.train()
    if hvd.rank() == 0:
        pbar = tqdm(train_loader)
    else:
        pbar = train_loader
    total_loss = 0.0
    total_map = 0.0
    for batch_idx, (data, target) in enumerate(pbar):
        if mixup:
            with torch.no_grad():
                data, target_a, target_b, lam = mixup_data(data, target, mixup_alpha)
                target_a, target_b = target_a.cuda(), target_b.cuda()

        data, target = data.cuda(), target.cuda()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
        output = model(data)
        if mixup:
            loss = lam * F.cross_entropy(output, target_a) + (1 - lam) * F.cross_entropy(output, target_b)
        else:
            loss = F.cross_entropy(output, target)

        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

        total_loss += loss
        total_loss = hvd.allreduce(total_loss)
        score = mapk(output, target)[0]
        total_map += score
        total_map = hvd.allreduce(total_map)

        if batch_idx == 0:
            mean_loss = total_loss
            mean_map = total_map

        mean_loss = mean_loss * 0.9 + 0.1 * (total_loss.item() / (batch_idx + 1))
        mean_map = mean_map * 0.9 + 0.1 * (total_map.item() / (batch_idx + 1))
        lr = optimizer.param_groups[0]['lr']
        if hvd.rank() == 0:
            pbar.set_description(f'Epoch: {epoch} Loss: {mean_loss:.4f} MAP@3: {mean_map*100.:.2f}% LR: {lr:.5f}')
    return checkpoint_path
