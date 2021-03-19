import onnxruntime
import numpy as np
import cv2
import torch

import torchvision
import torchvision.transforms as transforms
from models import *
from collections import OrderedDict



def get_classify_result(image,model,device):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    img_tensor = transform_test(image).unsqueeze(0)
    print(img_tensor)
#     print("img_tensor")
#     print(img_tensor)
#     print(img_tensor)
    img_tensor = img_tensor.permute(0, 2, 3, 1).to(device)
    print(img_tensor.size())
    print(img_tensor.dtype)
#     print(img_tensor.shape)
    with torch.no_grad():
        outputs = model(img_tensor)
#     print("outputs")
#     print(outputs)
    return outputs

image = cv2.imread("/home/liweiliang/project/pytorch-cifar/data/onnx_test/off/101.jpg")
# imgsz = 480
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(img[:5, :5, :])
# img = cv2.resize(img, (imgsz, imgsz))
img = cv2.resize(img, (32, 96))
# print(img[:5, :5, :] / 255.0)
# img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
# img = img.to(device, non_blocking=True)
# img = img.half() if half else img.float()  # uint8 to fp16/32
# img_detect = (img / 255.0).astype(np.float32).reshape(1,192,256,3)
# print(img_detect[:5, :5, :])
# input_array= np.ones([1,480,480,3]).astype(np.float32)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

img_tensor = transform_test(img)
img_array = img_tensor.unsqueeze(0).permute(0, 2, 3, 1).numpy()
print(img_array)
print(img_tensor)
# print(img_array)
print(img_array.shape)
sess = onnxruntime.InferenceSession('/home/liweiliang/project/pytorch-cifar/checkpoint/resnet.onnx')
input_nodes = sess.get_inputs()[0].name
for i in range(len(sess.get_outputs())):
    print(sess.get_outputs()[i].name)
class_data = sess.get_outputs()[0].name
print(class_data)
# print(input_array[0,0,0,0])
output = sess.run([   class_data], {input_nodes: img_array})
result  = output[0]
print(result.shape)
print("onnx result")
print(result)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ResNet18()
net = net.to(device)
checkpoint = torch.load('/home/liweiliang/project/pytorch-cifar/checkpoint/ckpt_190.pth')
new_state_dict = OrderedDict()
for k, v in checkpoint['net'].items():
#     print(k)
    name = k[7:]  # remove module.
    new_state_dict[name] = v
#     print(v)
net.load_state_dict(new_state_dict)
net.eval()
# image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = get_classify_result(img,net,device)
print("pytorch result")
print(result)