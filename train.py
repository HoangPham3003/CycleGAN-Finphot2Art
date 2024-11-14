import os
import cv2
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import down_data, ArtLoader
from model import Generator, Discriminator
from utils import weights_init, show_tensor_images
from losses import get_disc_loss, get_gen_loss


if __name__ == '__main__':
    
     ### =============== Hyper-parameters =============== ### 
    
    dim_A = 3
    dim_B = 3
    n_epochs = 20
    display_step = 200
    batch_size = 1
    lr = 0.0002
    load_shape = 286
    target_shape = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    adv_criterion = nn.MSELoss()
    recon_criterion = nn.L1Loss()
    
    save_model = False
    
    
    ### =============== Load data =============== ###
    
    ROOT_DIR = os.getcwd()
    
    DATA_DIR = down_data(ROOT_DIR)
    
    loader = ArtLoader(DATA_DIR=DATA_DIR, batch_size=batch_size, mode='train')
    train_loader = loader.load_data()
    
    print("Number of iterations: ", len(train_loader))
    
    
    ### =============== CycleGAN Model =============== ###
    
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
    
    
    ### =============== Training =============== ###
    
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    cur_step = 0
    
    for epoch in range(n_epochs):
        for real_A, real_B in tqdm(train_loader):
            real_A = nn.functional.interpolate(real_A, size=target_shape)
            real_B = nn.functional.interpolate(real_B, size=target_shape)
            cur_batch_size = len(real_A)
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            ### Update discriminator A ###
            disc_A_opt.zero_grad()
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)
            disc_A_loss.backward(retain_graph=True)
            disc_A_opt.step()
            
            ### Update discriminator B ###
            disc_B_opt.zero_grad()
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
            disc_B_loss.backward(retain_graph=True)
            disc_B_opt.step()
            
            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion
            )
            gen_loss.backward()
            gen_opt.step()
            
            mean_discriminator_loss += disc_A_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step
            
            ### Visualization code ###
            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                show_tensor_images(torch.cat([real_A, real_B]), size=(dim_A, target_shape, target_shape), current_step=cur_step, real=True)
                show_tensor_images(torch.cat([fake_B, fake_A]), size=(dim_B, target_shape, target_shape), current_step=cur_step, real=False)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    torch.save({
                        'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': disc_A_opt.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': disc_B_opt.state_dict()
                    }, f"CycleGAN_{cur_step}.pt")
            cur_step += 1