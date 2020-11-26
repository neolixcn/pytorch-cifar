'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--light_type', default ='q', type=str, help='traffic_light_type q--Quadrate,1:1;h--horizontal,1:3;v--vertical,3:1')
parser.add_argument('--data_root',type=str,help="data root")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

classes = ('red', 'green', 'yellow', 'off', 'others')
if args.light_type == 'q':
    h=64
    w=64
elif args.light_type == 'h':
    h = 32
    w=96
elif args.light_type == 'v':
    h = 96
    w = 32
else:
    print(" wrong type")
traffic_light_directory = '/nfs/nas/VOCdevkit/VOC2007_dtld_full/CLS_Images'
traffic_light_directory = args.data_root
transform_train = transforms.Compose([
        transforms.RandomCrop((w,h), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
# import cv2
# image= cv2.imread("/home/weiliang/work/data/TL/test_shanghai/datatest_front_12_png/resize/five_image/green/test_result_0_1.jpg")
#
# result = transform_test(image)
# print(result)
trainset = torchvision.datasets.ImageFolder(root=traffic_light_directory + '/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)
# testset = torchvision.datasets.ImageFolder(root=traffic_light_directory + '/no_padding/train', transform=transform_test)
testset = torchvision.datasets.ImageFolder(root=traffic_light_directory +'/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)

print(trainset.class_to_idx)

# Model/home/weiliang/work/neolix_perception/output/traffic_light_test_6mm/traffic_roi_generate//home/weiliang/work/neolix_perception/output/traffic_light_test_6mm/traffic_roi_generate/
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
#args.resume=True
#print(args.resume)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_v.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("get model done")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    total_each_class=[0, 0, 0, 0, 0]
    correct_each_class=[0, 0, 0, 0, 0]

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.permute(0, 2, 3, 1)
        # import pdb;pdb.set_trace()
        optimizer.zero_grad()

      
        outputs = net(inputs)
        # print(outputs)
        outputs = outputs.squeeze(0).squeeze(0)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        targets_cpu = targets.cpu().numpy()
        correct_cpu = predicted.eq(targets).cpu().numpy()

        for i, c in enumerate(total_each_class):
            total_each_class[i] += sum(targets_cpu==i)
        #print(total_each_class)

        for i, c in enumerate(correct_each_class):
            targets_cpu_each_class = targets_cpu==i
            predicted_cpu_each_class = correct_cpu
            correct_each_class[i] += sum(np.logical_and(targets_cpu_each_class, predicted_cpu_each_class))

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    for k, v in trainset.class_to_idx.items():
        print("%10s acc: %.3f"%((k), (correct_each_class[v]/total_each_class[v])))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    total_each_class=[0, 0, 0, 0, 0]
    correct_each_class=[0, 0, 0, 0, 0]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # if inputs.size()[0]<10:
            #     continue

            inputs, targets = inputs.to(device), targets.to(device)
            #print("input")
            #print(inputs.size())
            #print(inputs)
            inputs = inputs.permute(0, 2, 3, 1)
            outputs = net(inputs)
            outputs = outputs.squeeze(0).squeeze(0)
            #print("outputs")
            #print(outputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #print(predicted)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            targets_cpu = targets.cpu().numpy()
            correct_cpu = predicted.eq(targets).cpu().numpy()
            
            for i, c in enumerate(total_each_class):
                total_each_class[i] += sum(targets_cpu==i)
            #print(total_each_class)

            for i, c in enumerate(correct_each_class):
                targets_cpu_each_class = targets_cpu==i
                predicted_cpu_each_class = correct_cpu
                correct_each_class[i] += sum(np.logical_and(targets_cpu_each_class, predicted_cpu_each_class))

            #print(correct_each_class)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        for k, v in trainset.class_to_idx.items():
            print("%10s acc: %.3f"%((k), (correct_each_class[v]/total_each_class[v])))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
    # # if acc > 0:
        print('Saving......')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    # scheduler.step()
