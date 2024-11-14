import os
from tqdm.auto import tqdm

import torch

from datasets import down_data, ArtLoader
from model import Generator, Discriminator
from utils import weights_init

if __name__ == '__main__':
    # ROOT_DIR = os.getcwd()
    
    # DATA_DIR = down_data(ROOT_DIR)
    
    # loader = ArtLoader(DATA_DIR=DATA_DIR)
    # train_loader, test_loader = loader.load_data()
    
    # print(len(train_loader))
    # print(len(test_loader))
    
    dim_A = 3
    dim_B = 3
    lr = 0.0002
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    gen_AB = Generator(input_channels=dim_A, output_channels=dim_B).to(device)
    gen_BA = Generator(input_channels=dim_B, output_channels=dim_A).to(device)
    gen_opt = torch.optim.Adam(
        list(gen_AB.parameters()) + list(gen_BA.parameters()),
        lr=lr,
        betas=(0.5, 0.999)
    )
    
    disc_A = Discriminator(input_channels=dim_A).to(device)
    disc_A_opt = torch.optim.Adam(
        disc_A.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )
    
    disc_B = Discriminator(input_channels=dim_B).to(device)
    disc_B_opt = torch.optim.Adam(
        disc_B.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )
    
    gen_AB = gen_AB.apply(weights_init)
    gen_BA = gen_BA.apply(weights_init)
    disc_A = disc_A.apply(weights_init)
    disc_B = disc_B.apply(weights_init)
    
    print(disc_B)
    
    
    
    
    
    
        
    