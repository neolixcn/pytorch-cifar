import cv2
import os
import numpy as np
import torch
from models import *
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms

def get_classify_result(image,model,device):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    img_tensor = transform_test(image).unsqueeze(0)
#     print("img_tensor")
#     print(img_tensor)
#     print(img_tensor)
    img_tensor = img_tensor.permute(0, 2, 3, 1).to(device)
#     print(img_tensor.size())
#     print(img_tensor.dtype)
#     print(img_tensor.shape)
    print(img_tensor.size())
    with torch.no_grad():
        outputs = model(img_tensor)
#     print("outputs")
#     print(outputs)
    return outputs


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ResNet18()
net = net.to(device)
checkpoint = torch.load('./checkpoint/ckpt.pth',map_location=device)
new_state_dict = OrderedDict()
for k, v in checkpoint['net'].items():
#     print(k)
    name = k[7:]  # remove module.
    new_state_dict[name] = v
#     print(v)
net.load_state_dict(new_state_dict)
net.eval()
color_list = ["green","off","red","yellow"]



for file in os.listdir("./data/test/yellow/"):
    image_ori = cv2.imread("./data/test/yellow/"+file)
    image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    result = get_classify_result(image,net,device)
    # import pdb;pdb.set_trace()
    print(result[0])
    _,color_index=result.max(1)
    result_index = color_index.cpu().numpy()[0]
    class_name = color_list[result_index]
    cv2.imwrite(f"./{class_name}_18_"+file,image_ori)