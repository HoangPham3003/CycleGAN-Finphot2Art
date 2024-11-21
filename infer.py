import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import Generator


def infer(pretrained_path, image_path, infer_transforms_, save_dir, device):
    gen_BA = Generator(input_channels=3, output_channels=3).to(device)
    
    print(pretrained_path)
    if not os.path.exists(pretrained_path):
        print("Pretrained model does not exist!")
        return
    
    print("Loading pretrained...")
    pretrained_model = torch.load(pretrained_path, weights_only=True)
    gen_BA.load_state_dict(pretrained_model['gen_BA'])
    print("Pretrained model is ready!")
    
    transform = transforms.Compose(infer_transforms_)
    image = Image.open(image_path)
    image = transform(image)
    image = image.to(device)
    
    with torch.no_grad():
        fake_A = gen_BA(image)
    output_fake = ((fake_A + 1) / 2) * 255
    output_fake = output_fake.cpu().detach().numpy()
    output_fake = output_fake.transpose(1, 2, 0)
    output_fake = cv2.cvtColor(output_fake, cv2.COLOR_BGR2RGB)
    
    image_name = image_path.split('/')[-1].split('.')[0]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = f"{save_dir}/infer_{image_name}.jpg"
    work_dir = os.getcwd()
    save_path = os.path.join(work_dir, save_path)
    cv2.imwrite(save_path, output_fake)
    print("The result is saved at : {}".format(save_path))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pretrained", type=str, default="pretrained/CycleGAN.pt", help="name of the pretrained model")
    parser.add_argument("-ip", "--image_path", type=str, default="DATA_INFER/img1.jpg", help="infered image path")
    parser.add_argument("-sd", "--save_dir", type=str, default="inference", help="name of save folder")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()
    
    infer_transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    infer(args.pretrained, args.image_path, infer_transforms_, args.save_dir, args.device)
    
    
    
    
    
    
    

    
        
    
    
    
    
    