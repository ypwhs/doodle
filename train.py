import torch
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils as utils

import horovod.torch as hvd
from dataset import get_split_dataloader
from training import train, test, save_checkpoint, load_checkpoint, change_lr
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import argparse

hvd.init()
torch.cuda.set_device(hvd.local_rank())

parser = argparse.ArgumentParser(description='doodle training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', default='resnet34', type=str, help='model_name')
parser.add_argument('--tag', default='test', type=str, help='model_name')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint path')
parser.add_argument('--width', default=256, type=int, help='strokes image width')
parser.add_argument('--batch_size', default=224, type=int, help='batch_size')
parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
parser.add_argument('--nowarmup', action='store_true', help='skip warmup')

args = parser.parse_args()

if hvd.rank() == 0:
    print(args)


model = pretrainedmodels.__dict__[args.model](num_classes=1000, pretrained='imagenet')
model.name = f'{args.model}_{args.tag}'

transform = utils.TransformImage(model)
width = args.width
batch_size = args.batch_size
num_workers = args.num_workers
num_classes = 340
evaluate_interval = 10

model.last_linear = nn.Linear(in_features=model.last_linear.in_features, out_features=num_classes, bias=True)
model = model.cuda()

valid_loader = get_split_dataloader(f'split_recognized/train_k99.csv', width=width, batch_size=batch_size,
                                    transform=transform, num_workers=num_workers)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

scale_lr = batch_size * hvd.size() / 128
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

scheduler_warmup = LambdaLR(optimizer, lambda step: 1 + (scale_lr - 1) * step / len(valid_loader) / 5)
scheduler_train = MultiStepLR(optimizer, milestones=[30, 60, 80])
scheduler_train.base_lrs = [x * scale_lr for x in scheduler_train.base_lrs]

if args.checkpoint:
    load_checkpoint(model, optimizer, args.checkpoint)

hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

if not args.nowarmup:
    for i in range(5):
        epoch = i + 1
        train_loader = get_split_dataloader(f'split_recognized/train_k{i}.csv', width=width, batch_size=batch_size,
                                            transform=transform, num_workers=num_workers)
        train(model, train_loader, optimizer=optimizer, epoch=epoch, scheduler=scheduler_warmup)

    mean_loss, mean_map, t = test(model, valid_loader)
    if hvd.rank() == 0:
        checkpoint_path = save_checkpoint(model, optimizer, test_acc=mean_map, tag='warmup')

# training
for i in range(5, 99):
    epoch = i + 1
    train_loader = get_split_dataloader(f'split_recognized/train_k{i}.csv', width=width, batch_size=batch_size,
                                        transform=transform, num_workers=num_workers)
    train(model, train_loader, optimizer=optimizer, epoch=epoch)
    if epoch % evaluate_interval == 0:
        mean_loss, mean_map, t = test(model, valid_loader)
        if hvd.rank() == 0:
            checkpoint_path = save_checkpoint(model, optimizer, test_acc=mean_map, tag=epoch)
