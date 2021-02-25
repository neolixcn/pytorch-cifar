import os 
import numpy as np
import cv2

root_path = "/home/liweiliang/project/pytorch-cifar/data/v/train/off/"
save_path = "/home/liweiliang/project/pytorch-cifar/data/h/train/off/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
for file_name in os.listdir(root_path):
    image = cv2.imread(root_path +file_name)
    image_h = np.rot90(image,1)

    cv2.imwrite(save_path+file_name,image_h,)
