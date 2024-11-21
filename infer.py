import os
import cv2
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
    pretrained_model = torch.load(pretrained_path)
    gen_BA.load_state_dict(pretrained_model['gen_BA'])
    print("Pretrained model is ready!")
    
    transform = transforms.Compose(infer_transforms_)
    image = Image.open(image_path)
    image = transform(image)
    image = nn.functional.interpolate(image, size=256)
    
    image = image.to(device)
    
    with torch.no_grad():
        fake_A = gen_BA(image)
    output_fake = fake_A.cpu().detach().numpy()
    plt.imshow(output_fake)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(f"{save_dir}/infer.jpg")
    print("The result is saved at : {}".format(f"{save_dir}/infer.jpg"))
    
    
if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    work_dir = os.getcwd()
    save_dir = f"{work_dir}/inference"
    
    image_dir = f"{work_dir}/DATA_INFER"
    pretrained_dir = f"{work_dir}/pretrained"
    
    pretrained_path = f"{pretrained_dir}/CycleGAN.pt"
    print(pretrained_path)
    
    infer_transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    image_path = f"{image_dir}/img1.jpg"
    
    infer(pretrained_path, image_path, infer_transforms_, save_dir, device)
    
    
    
    
    
    
    

    
        
    
    
    
    
    