import torch
import torch.nn as nn
import pretrainedmodels
import torchvision.transforms as transforms

import horovod.torch as hvd
from dataset import get_split_dataloader
from training import train, test, save_checkpoint, load_checkpoint
import argparse

hvd.init()
torch.cuda.set_device(hvd.local_rank())

parser = argparse.ArgumentParser(description='doodle training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', default='resnet34', type=str, help='model_name')
parser.add_argument('--tag', default='test', type=str, help='model_name')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint path')
parser.add_argument('--width', default=256, type=int, help='strokes image width')
parser.add_argument('--batch_size', default=512, type=int, help='batch_size')
parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()

if hvd.rank() == 0:
    print(args)

width = args.width
batch_size = args.batch_size
num_workers = args.num_workers
num_classes = 340
evaluate_interval = 10

model = pretrainedmodels.__dict__[args.model](num_classes=1000, pretrained=None)
model.name = f'{args.model}_{args.tag}'
model.avgpool = nn.AdaptiveAvgPool2d(1)
model.last_linear = nn.Linear(in_features=model.last_linear.in_features, out_features=num_classes, bias=True)

transform = transforms.Compose([
    transforms.Resize(size=(width, width)),
    transforms.ToTensor()
])

model = model.cuda()

valid_loader = get_split_dataloader(f'split_recognized/train_k99.csv', batch_size=batch_size,
                                    transform=transform, num_workers=num_workers)

scale_lr = batch_size * hvd.size() / 128
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * scale_lr, amsgrad=True)
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

epoch = 0
if args.checkpoint:
    load_checkpoint(model, optimizer, args.checkpoint)
    tag = args.checkpoint.split('_')[-1].split('.')[0]
    if tag.isnumeric():
        epoch = int(tag)

hvd.broadcast_parameters(model.state_dict(), root_rank=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5623)

# training
for i in range(epoch):
    scheduler.step()

for i in range(epoch, 400):
    epoch += 1
    scheduler.step()
    train_loader = get_split_dataloader(f'split_recognized/train_k{i % 99}.csv', width=width, batch_size=batch_size,
                                        transform=transform, num_workers=num_workers)
    train(model, train_loader, optimizer=optimizer, epoch=epoch)
    if epoch % evaluate_interval == 0:
        mean_loss, mean_map, t = test(model, valid_loader)
        if hvd.rank() == 0:
            checkpoint_path = save_checkpoint(model, optimizer, test_acc=mean_map, tag=epoch)

mean_loss, mean_map, t = test(model, valid_loader)
if hvd.rank() == 0:
    checkpoint_path = save_checkpoint(model, optimizer, test_acc=mean_map, tag=epoch)
