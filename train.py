import torch
import torch.nn as nn
import pretrainedmodels
import torchvision.transforms as transforms

import argparse
import horovod.torch as hvd
from dataset import get_split_dataloader
from training import train, test, save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from WideResNeXt import get_wideresnext34
from fp16util import network_to_half

hvd.init()
torch.cuda.set_device(hvd.local_rank())

parser = argparse.ArgumentParser(description='doodle training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', default='resnet34', type=str, help='model_name')
parser.add_argument('--tag', default='test', type=str, help='model_name')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint path')
parser.add_argument('--width', default=256, type=int, help='strokes image width')
parser.add_argument('--batch_size', default=224, type=int, help='batch_size')
parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
parser.add_argument('--dataset', default='split', type=str, help='dataset')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--pretrained', action='store_true', help='use imagenet pretrained weights')
parser.add_argument('--half', action='store_true', help='half')
args = parser.parse_args()

if hvd.rank() == 0:
    print(args)

if args.model == 'wideresnext34':
    model = get_wideresnext34()
else:
    model = pretrainedmodels.__dict__[args.model](num_classes=1000, pretrained='imagenet' if args.pretrained else None)

width = args.width
batch_size = args.batch_size
num_workers = args.num_workers
num_classes = 340
evaluate_interval = 10

model.avgpool = nn.AdaptiveAvgPool2d(1)
model.avg_pool = nn.AdaptiveAvgPool2d(1)
model.avgpool_1a = nn.AdaptiveAvgPool2d(1)
if args.model == 'wideresnext34':
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
else:
    model.last_linear = nn.Linear(in_features=model.last_linear.in_features, out_features=num_classes, bias=True)

transform = transforms.Compose([
    transforms.Resize(size=(width, width)),
    transforms.ToTensor()
])

valid_loader = get_split_dataloader(f'{args.dataset}/train_k99.csv', width=width, batch_size=batch_size,
                                    transform=transform, num_workers=num_workers)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

epoch = 0
if args.checkpoint:
    load_checkpoint(model, optimizer, args.checkpoint)
    tag = args.checkpoint.split('_')[-1].split('.')[0]
    if tag == 'warmup':
        epoch = 5
    elif tag.isnumeric():
        epoch = int(tag)

if args.half:
    model = network_to_half(model)
model = model.cuda()
model.name = f'{args.model}_{args.tag}'

hvd.broadcast_parameters(model.state_dict(), root_rank=0)

scale_lr = batch_size * hvd.size() / 128
scheduler_warmup = LambdaLR(optimizer, lambda step: 1 + (scale_lr - 1) * step / len(valid_loader) / 5)
scheduler_train = MultiStepLR(optimizer, milestones=[80, 160, 200])
scheduler_train.base_lrs = [x * scale_lr for x in scheduler_train.base_lrs]

if epoch == 0:
    for i in [20, 40, 60, 80, 99]:
        epoch += 1
        train_loader = get_split_dataloader(f'{args.dataset}/train_k{i}.csv', width=width, batch_size=batch_size,
                                            transform=transform, num_workers=num_workers)
        train(model, train_loader, optimizer=optimizer, epoch=epoch, scheduler=scheduler_warmup, half=args.half)

    mean_loss, mean_map, t = test(model, valid_loader, half=args.half)
    if hvd.rank() == 0:
        checkpoint_path = save_checkpoint(model, optimizer, test_acc=mean_map, tag='warmup')

# training
for i in range(epoch):
    scheduler_train.step()

for i in range(epoch, 240):
    epoch += 1
    scheduler_train.step()
    train_loader = get_split_dataloader(f'{args.dataset}/train_k{i % 99}.csv', width=width, batch_size=batch_size,
                                        transform=transform, num_workers=num_workers)
    train(model, train_loader, optimizer=optimizer, epoch=epoch, half=args.half)
    if epoch % evaluate_interval == 0:
        mean_loss, mean_map, t = test(model, valid_loader, half=args.half)
        if hvd.rank() == 0:
            checkpoint_path = save_checkpoint(model, optimizer, test_acc=mean_map, tag=epoch)

mean_loss, mean_map, t = test(model, valid_loader, half=args.half)
if hvd.rank() == 0:
    checkpoint_path = save_checkpoint(model, optimizer, test_acc=mean_map, tag=epoch)
