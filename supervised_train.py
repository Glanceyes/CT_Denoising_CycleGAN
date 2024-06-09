import torch
import torch.nn as nn
import torchvision
import random
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from os import makedirs
from os.path import isdir, join 
from torch.utils.data import  DataLoader
from torch.nn import init
from tqdm.auto import tqdm

from cycleGAN_train import CT_Dataset, Generator, Mean, init_weights
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Functions for caculating PSNR, SSIM
# Peak Signal-to-Noise Ratio
def psnr(A, ref):
    ref[ref < -1000] = -1000
    A[A < -1000] = -1000
    val_min = -1000
    val_max = np.amax(ref)
    ref = (ref - val_min) / (val_max - val_min)
    A = (A - val_min) / (val_max - val_min)
    out = peak_signal_noise_ratio(ref, A)
    return out

# Structural similarity index
def ssim(A, ref):
    ref[ref < -1000] = -1000
    A[A < -1000] = -1000
    val_min = -1000
    val_max = np.amax(ref)
    ref = (ref - val_min) / (val_max - val_min)
    A = (A - val_min) / (val_max - val_min)
    out = structural_similarity(ref, A, data_range=2)
    return out


# Make dataloader for training/test
def make_dataloader(path, train_batch_size=1, is_train=True):
    # Path of 'train' and 'test' folders    
    dataset_path = join(path, 'train') if is_train else join(path, 'test')

    # Transform for training data: convert to tensor, random horizontal/verical flip, random crop
    # You can change transform if you want.
    if is_train:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        train_dataset = CT_Dataset(dataset_path, train_transform, shuffle=False)
        dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    else:
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        test_dataset = CT_Dataset(dataset_path, test_transform, shuffle=False)
        dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    return dataloader


def train(
    path_checkpoint='./CT_denoising',
    model_name='supervised_v1',
    path_data='../data/AAPM_data',
    batch_size=16,
    beta1=0.5,
    beta2=0.999,
    num_epoch=100,
    g_channels=32,
    ch_mult=[1, 2, 4, 8],
    num_res_blocks=3,
    lr=5e-4,
    use_checkpoint = False
):
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path for saving the checkpoint
    if not isdir(path_checkpoint):
        makedirs(path_checkpoint)

    # Path for saving results
    path_result = join(path_checkpoint, model_name)
    if not isdir(path_result):
        makedirs(path_result)

    # Make dataloaders
    train_dataloader = make_dataloader(path_data, batch_size)

    G_Q2F = Generator(1, 1, g_channels, ch_mult=ch_mult, num_res_blocks=num_res_blocks).to(device)

    # Make optimizers
    G_optim = torch.optim.Adam(G_Q2F.parameters(), lr, betas=(beta1, beta2))
   
    # Define loss functions
    supervised_loss = nn.L1Loss()

    # Loss functions
    loss_name = ['G_supervised_loss']

    if use_checkpoint:
        # If a checkpoint exists, load the state of the model and optimizer from the checkpoint
        checkpoint = torch.load(join(path_checkpoint, model_name + '.pth'))
        G_Q2F.load_state_dict(checkpoint['G_Q2F_state_dict'])
    else:
        init_weights(G_Q2F)
    
    # Set the initial trained epoch as 0
    trained_epoch = 0
    
    # Initialize a dictionary to store the losses
    losses_list = {name: list() for name in loss_name}
    print('Start from random initialized model')

    # Start the training loop
    for epoch in tqdm(range(trained_epoch, num_epoch), desc='Epoch', total=num_epoch, initial=trained_epoch):
        # Initialize a dictionary to store the mean losses for this epoch
        losses = {name: Mean() for name in loss_name}

        for x_F, x_Q, _ in tqdm(train_dataloader, desc='Step'):
            # Move the data to the device (GPU or CPU)
            x_F = x_F.to(device)
            x_Q = x_Q.to(device)

            # Denoise the quarter-dose image
            x_QF = G_Q2F(x_Q)

            # Calculate the supervised loss
            G_supervised_loss = supervised_loss(x_QF, x_F)
            
            # Update the generators
            G_optim.zero_grad()
            G_supervised_loss.backward()
            G_optim.step()

            # Calculate the supervised loss during one epoch
            losses['G_supervised_loss'](G_supervised_loss.detach())
    
        for name in loss_name:
            losses_list[name].append(losses[name].result())
        
        # Save the trained model and list of losses
        torch.save({'epoch': epoch + 1, 'G_Q2F_state_dict': G_Q2F.state_dict(),}, join(path_checkpoint, model_name + '.pth'))
        for name in loss_name:
            torch.save(losses_list[name], join(path_result, name + '.npy'))
    
    plt.figure(1)
    for name in ['G_supervised_loss']:
        loss_arr = torch.load(join(path_result, name + '.npy'), map_location='cpu')
        x_axis = np.arange(1, len(loss_arr) + 1)
        plt.plot(x_axis, loss_arr, label=name)
        
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(join(path_result, 'loss_curve.png'))
    plt.close() 

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_checkpoint', type=str, default='./CT_denoising')
    parser.add_argument('--model_name', type=str, default='supervised_v1')
    parser.add_argument('--path_data', type=str, default='./AAPM_data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_epoch', type=int, default=120)
    parser.add_argument('--g_channels', type=int, default=32)
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num_res_blocks', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_checkpoint', action='store_true')
    
    args = parser.parse_args()
    
    train(
        path_checkpoint=args.path_checkpoint,
        model_name=args.model_name,
        path_data=args.path_data,
        batch_size=args.batch_size,
        beta1=args.beta1,
        beta2=args.beta2,
        num_epoch=args.num_epoch,
        g_channels=args.g_channels,
        ch_mult=args.ch_mult,
        num_res_blocks=args.num_res_blocks,
        lr=args.lr,
        use_checkpoint=args.use_checkpoint
    )