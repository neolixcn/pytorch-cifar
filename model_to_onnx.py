import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.nn as nn

from torch.utils import data
from tqdm import tqdm

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
# parser = argparse.ArgumentParser(description="config")
# parser.add_argument(
#     "--config",
#     nargs="?",
#     type=str,
#     default="configs/hardnet.yml",
#     help="Configuration file to use",
# )
#
# args = parser.parse_args()

# with open(args.config) as fp:
#     cfg = yaml.load(fp)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# model = get_model(cfg["model"], 19).to(device)
# state = convert_state_dict(torch.load("/home/weiliang/work/git_code/FCHarDNet-master/weights/hardnet70_cityscapes_model.pkl", map_location=device)["model_state"])
# model.load_state_dict(state)
# model.eval()
# model.to(device)

# inputs = torch.ones(1, 3, 720, 1280)
inputs = torch.ones(1,  96, 32,3)
# img = cv2
# import cv2
# inputs = cv2.imread("/home/weiliang/work/git_code/FCHarDNet-master/test_image/971.jpg")
# inputs = torch.ones(1, 3, 540, 960)
with torch.no_grad():
    # features, regression, classification, anchors = model(x)
    # import time
    # value_scale = 255
    # mean = [0.406, 0.456, 0.485]
    # mean = [item * value_scale for item in mean]
    # std = [0.225, 0.224, 0.229]
    # std = [item * value_scale for item in std]
    # img = (inputs - mean) / std
    #
    # img = img.transpose(2, 0, 1)
    # img = np.expand_dims(img, 0)
    # img = torch.from_numpy(img).float()
    #
    #
    # images = img.to(device)


    # for i in range(1):
    #     start = time.time()
    #
    #     # images = img.to(device)
    #
    #     output = model(images)
    #     end = time.time()
    #     print(f"use time {(end - start) * 1000} ms")
    #     # time.sleep(1)

    net = ResNet18()
    net = net.to(device)
    # if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True
    checkpoint = torch.load('./checkpoint/ckpt_v_003.pth')


    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        print(k)
        name = k[7:]  # remove module.
        new_state_dict[name] = v
        print(v)

    net.load_state_dict(new_state_dict)
    net.eval()

    inputs = inputs.to(device)
    torch.onnx.export(net, inputs,
                      f'./checkpoint/rsnet_003.onnx',
                      verbose=True,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version= 10,
                      # dynamic_axes={"input": {0: "batch_size"},
                      #               "output": {0: "batch_size"}}
                      )
