
import random
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from os import listdir, makedirs
from os.path import isdir, join 
from torch.utils.data import Dataset, DataLoader
from torch.nn import init

from tqdm.auto import tqdm
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

# Initialize parameters of neural networks
def init_weights(net):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        
    print('Initialize network.')
    net.apply(init_func)
    
    
# Set 'requires_grad' of the networks
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# Calculate average loss during one epoch
class Mean:
    def __init__(self):
        self.numel = 0
        self.mean = 0
    
    def __call__(self, val):
        self.mean = self.mean * (self.numel / (self.numel + 1)) + val / (self.numel + 1)
        self.numel += 1
    
    def result(self):
        return self.mean


# CT dataset
class CT_Dataset(Dataset):
    def __init__(self, path, transform, shuffle=True):
        # Path of 'full_dose' and 'quarter_dose' folders
        self.path_full = join(path, 'full_dose')
        self.path_quarter = join(path, 'quarter_dose')
        self.transform = transform

        # File list of full dose data
        self.file_full = list()
        for file_name in sorted(listdir(self.path_full)):
            self.file_full.append(file_name)
            
        if shuffle:
            random.seed(0)
            random.shuffle(self.file_full)
        
        # File list of quarter dose data
        self.file_quarter = list()
        for file_name in sorted(listdir(self.path_quarter)):
            self.file_quarter.append(file_name)
    
    def __len__(self):
        return min(len(self.file_full), len(self.file_quarter))
    
    def __getitem__(self, idx):
        # Load full dose/quarter dose data
        x_F = np.load(join(self.path_full, self.file_full[idx]))
        x_Q = np.load(join(self.path_quarter, self.file_quarter[idx]))

        # Convert to HU scale
        x_F = (x_F - 0.0192) / 0.0192 * 1000
        x_Q = (x_Q - 0.0192) / 0.0192 * 1000

        # Normalize images
        x_F[x_F < -1000] = -1000
        x_Q[x_Q < -1000] = -1000

        x_F = x_F / 4000
        x_Q = x_Q / 4000

        # Apply transform
        x_F = self.transform(x_F)
        x_Q = self.transform(x_Q)

        file_name = self.file_quarter[idx]

        return x_F, x_Q, file_name
  

# Transform for the random crop
class RandomCrop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size
    
    def __call__(self, img):
        # Randomly crop the image into a patch with the size [self.patch_size, self.patch_size]
        w, h = img.size(-1), img.size(-2)
        i = random.randint(0, h - self.patch_size)
        j = random.randint(0, w - self.patch_size)

        return img[:, i:i + self.patch_size, j:j + self.patch_size]


# Make dataloader for training/test
def make_dataloader(path, train_batch_size=1, is_train=True):
    # Path of 'train' and 'test' folders    
    dataset_path = join(path, 'train') if is_train else join(path, 'test')

    # Transform for training data: convert to tensor, random horizontal/verical flip, random crop
    if is_train:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            RandomCrop(128)
        ])
        train_dataset = CT_Dataset(dataset_path, train_transform)
        dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    else:
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        test_dataset = CT_Dataset(dataset_path, test_transform)
        dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    return dataloader

  

class ResnetBlock(nn.Module):
    '''
    Residual block
    
    This class represents a residual block in a ResNet architecture. It consists of two convolutional layers
    with batch normalization and ReLU activation functions, and a shortcut connection to handle the case when
    the input and output channels are different.
    
    Args:
        in_channels (int): The number of input channels.
        out_channels (int, optional): The number of output channels. If not specified, it is set to the same
            as the input channels.
        dropout (float, optional): The dropout rate. Default is 0.5.
        num_groups (int, optional): The number of groups to separate the channels into for group normalization.
            Default is 16.
    '''
    def __init__(self, in_channels, out_channels=None, num_groups=16):
        super(ResnetBlock, self).__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # Group normalization layer and convolutional layer
        self.norm1 = torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = torch.nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Shortcut connection
        self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.relu(h, inplace=True)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x + h


class Upsample(nn.Module):
    '''
    Upsample module that performs bilinear upsampling followed by convolution, batch normalization, and activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int, optional): Number of output channels. If not provided, it will be set to the same as in_channels.

    Attributes:
        up (torch.nn.Upsample): Bilinear upsampling layer.
        conv (torch.nn.Conv2d): Convolutional layer.
        norm (torch.nn.BatchNorm2d): Batch normalization layer.
    '''

    def __init__(self, in_channels, out_channels=None):
        super(Upsample, self).__init__()
        if not out_channels:
            # If out_channels is not provided, set it to the same as in_channels
            out_channels = in_channels
        # Bilinear upsampling layer
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Convolutional layer
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Batch normalization layer
        self.norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        '''
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        '''
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        # Leaky ReLU activation function
        x = torch.nn.LeakyReLU(0.1)(x)
        return x


class Downsample(nn.Module):
    '''
    A class representing a downsampling module.

    This module performs downsampling on the input tensor using a convolutional layer with a stride of 2.

    Args:
        in_channels (int): The number of input channels.

    Attributes:
        in_channels (int): The number of input channels.
        conv (torch.nn.Conv2d): The convolutional layer used for downsampling.
    '''

    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.in_channels = in_channels
        self.conv = torch.nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        '''
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying convolution.

        '''
        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)

        return x

class Generator(nn.Module):
    '''
    Generator class for the CycleGAN model.
    
    Args:
        in_channels (int): The number of channels of the input.
        out_channels (int): The number of channels of the output.
        ngf (int): The number of convolution filters of the first layer.
        ch_mult (tuple): The channel multiplier for each resolution level. Default is (1, 2, 4, 8).
        num_res_blocks (int): The number of residual blocks in each resolution level. Default is 3.
    '''
    
    def __init__(self, in_channels, out_channels, ngf, ch_mult=(1, 2, 4, 8), num_res_blocks=3):
        super(Generator, self).__init__()
        
        # Check if the number of input channels is equal to the number of output channels
        assert in_channels == out_channels, 'The number of input channels should be equal to the number of output channels.'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = ngf
        
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # Initialize the lists of upsample and downsample blocks
        self.up_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.mid_block = nn.Module()

        # The first layer of the generator
        self.conv_in = nn.Conv2d(in_channels, ngf, kernel_size=3, stride=1, padding=1)
    
        # Initialize the number of input channels for each resolution level
        in_ch_mult = (1,) + tuple(ch_mult)

        # Define the downsample and upsample blocks
        for level in range(self.num_resolutions):
            down_block = nn.ModuleList()
            # The number of input and output channels for the current block
            block_in_channels = ngf * in_ch_mult[level]
            block_out_channels = ngf * ch_mult[level]

            for _ in range(self.num_res_blocks):
                # Add a residual block to the downsample block
                down_block.append(ResnetBlock(block_in_channels, block_out_channels))
                block_in_channels = block_out_channels
        
            if level != self.num_resolutions - 1:
                # Add a downsample block to the downsample blocks list
                down_block.append(Downsample(block_out_channels))

            self.down_blocks.append(down_block)

        # The middle block of the generator
        self.mid_block = ResnetBlock(ngf * ch_mult[-1], ngf * ch_mult[-1])

        for level in reversed(range(self.num_resolutions)):
            up_block = nn.ModuleList()
            # The number of input and output channels for the current block
            block_in_channels = ngf * ch_mult[level]
            block_out_channels = ngf * ch_mult[level]
            block_skip_channels = ngf * ch_mult[level]

            for block_idx in range(self.num_res_blocks + 1):
                if block_idx == self.num_res_blocks:
                    # If this is the last block, add a residual block with skip connections
                    block_skip_channels = ngf * in_ch_mult[level]
                    block_out_channels = ngf * in_ch_mult[level]
                # Add a residual block to the upsample block
                up_block.append(ResnetBlock(block_in_channels + block_skip_channels, block_out_channels))
                block_in_channels = block_out_channels

            if level != 0:
                up_block.append(Upsample(block_out_channels))
            
            self.up_blocks.insert(0, up_block)

        self.conv_out = torch.nn.Conv2d(ngf, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        '''
        Forward pass of the CycleGAN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        '''
        # Store the input tensor as the skip connection
        hs = [self.conv_in(x)]

        # Pass the input tensor through the downsample blocks
        for level in range(self.num_resolutions):
            for block in self.down_blocks[level]:
                h = block(hs[-1])
                # Store the output tensor of the residual block to the skip connections list
                hs.append(h)

        h = self.mid_block(hs[-1])

        # Pass the input tensor through the upsample blocks
        for level in reversed(range(self.num_resolutions)):
            for block in self.up_blocks[level]:
                if not isinstance(block, Upsample):
                    # If the block is not an upsample block, concatenate the skip connection
                    h = torch.cat([h, hs.pop()], dim=1)
                h = block(h)
        
        h = self.conv_out(h)
        h = h + x
        
        return h
  
# Discriminator (PatchGAN)
class Discriminator(nn.Module):
    '''
    Discriminator network for CycleGAN.

    Args:
        in_channels (int): Number of input channels.
        ndf (int): Number of discriminator filters.

    Attributes:
        in_channels (int): Number of input channels.
        ndf (int): Number of discriminator filters.
        conv1 (nn.Conv2d): Convolutional layer 1.
        conv2 (nn.Conv2d): Convolutional layer 2.
        conv3 (nn.Conv2d): Convolutional layer 3.
        conv4 (nn.Conv2d): Convolutional layer 4.
        conv5 (nn.Conv2d): Convolutional layer 5.

    '''

    def __init__(self, in_channels, ndf=32):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.ndf = ndf

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1)
        self.conv5 = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1)
    
    def forward(self, x, threshold=0.2):
        '''
        Forward pass of the discriminator network.

        Args:
            x (torch.Tensor): Input tensor.
            threshold (float): Leaky ReLU threshold.

        Returns:
            torch.Tensor: Output tensor.

        '''
        h = self.conv1(x)
        h = nn.functional.leaky_relu(h, threshold)

        h = self.conv2(h)
        h = nn.InstanceNorm2d(h.size(1))(h)
        h = nn.functional.leaky_relu(h, threshold)

        h = self.conv3(h)
        h = nn.InstanceNorm2d(h.size(1))(h)
        h = nn.functional.leaky_relu(h, threshold)

        h = self.conv4(h)
        h = nn.InstanceNorm2d(h.size(1))(h)
        h = nn.functional.leaky_relu(h, threshold)

        h = self.conv5(h)
        return h


# Training function
def train(
    path_checkpoint='./CT_denoising',
    model_name='cyclegan_v1',
    path_data='../data/AAPM_data',
    batch_size=16,
    lambda_cycle=10,
    lambda_iden=5,
    beta1=0.5,
    beta2=0.999,
    num_epoch=100,
    g_channels=32,
    d_channels=64,
    ch_mult=[1, 2, 4, 8],
    num_res_blocks=3,
    lr=2e-4,
    use_checkpoint=False
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

    # Make generators (G_F2Q: full to quarter / G_Q2F: quarter to full)
    G_F2Q = Generator(1, 1, g_channels, ch_mult=ch_mult, num_res_blocks=num_res_blocks).to(device)
    G_Q2F = Generator(1, 1, g_channels, ch_mult=ch_mult, num_res_blocks=num_res_blocks).to(device)

    # Make discriminators (D_F: distinguish real/fake full dose images / D_Q: distinguish real/fake quarter dose images)
    D_F = Discriminator(1, d_channels).to(device)
    D_Q = Discriminator(1, d_channels).to(device)

    # Make optimizers
    G_optim = torch.optim.Adam(itertools.chain(G_F2Q.parameters(), G_Q2F.parameters()), lr, betas=(beta1, beta2))
    D_optim = torch.optim.Adam(itertools.chain(D_F.parameters(), D_Q.parameters()), lr, betas=(beta1, beta2))

    # Define loss functions
    adv_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    iden_loss = nn.L1Loss()

    # Loss functions
    loss_name = ['G_adv_loss_F',
                'G_adv_loss_Q',
                'G_cycle_loss_F',
                'G_cycle_loss_Q',
                'G_iden_loss_F',
                'G_iden_loss_Q',
                'D_adv_loss_F',
                'D_adv_loss_Q']

    if use_checkpoint:
        # If a checkpoint exists, load the state of the model and optimizer from the checkpoint
        checkpoint = torch.load(join(path_checkpoint, model_name + '.pth'))
        G_Q2F.load_state_dict(checkpoint['G_Q2F_state_dict'])
        G_F2Q.load_state_dict(checkpoint['G_F2Q_state_dict'])
        D_Q.load_state_dict(checkpoint['D_Q_state_dict'])
        D_F.load_state_dict(checkpoint['D_F_state_dict'])
        G_optim.load_state_dict(checkpoint['G_optim_state_dict'])
        D_optim.load_state_dict(checkpoint['D_optim_state_dict'])
    else:
        # If no checkpoint exists, initialize the weights of the models
        init_weights(G_F2Q)
        init_weights(G_Q2F)
        init_weights(D_F)
        init_weights(D_Q)
        
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

            # Set 'requires_grad' of the discriminators as 'False' to avoid computing gradients of the discriminators
            set_requires_grad([D_F, D_Q], False)

            # Generate fake images using the generators
            x_FQ = G_F2Q(x_F)
            x_QF = G_Q2F(x_Q)
            
            # Generate cyclic images using the generators
            x_QFQ = G_F2Q(x_QF)
            x_FQF = G_Q2F(x_FQ)
            
            # Generate identity images using the generators
            x_QQ = G_F2Q(x_Q)
            x_FF = G_Q2F(x_F)

            # Calculate adversarial losses
            G_adv_loss_F = adv_loss(D_F(x_QF), torch.ones_like(D_F(x_QF)))
            G_adv_loss_Q = adv_loss(D_Q(x_FQ), torch.ones_like(D_Q(x_FQ)))
            
            # Calculate cycle losses
            G_cycle_loss_F = cycle_loss(x_FQF, x_F)
            G_cycle_loss_Q = cycle_loss(x_QFQ, x_Q)
            
            # Calculate identity losses
            G_iden_loss_F = iden_loss(x_FF, x_F)
            G_iden_loss_Q = iden_loss(x_QQ, x_Q)
            
            # Calculate total losses
            G_adv_loss = G_adv_loss_F + G_adv_loss_Q
            G_cycle_loss = G_cycle_loss_F + G_cycle_loss_Q
            G_iden_loss = G_iden_loss_F + G_iden_loss_Q
            G_total_loss = G_adv_loss + lambda_cycle * (G_cycle_loss) + lambda_iden * (G_iden_loss)

            # Update the generators
            G_optim.zero_grad()
            G_total_loss.backward()
            G_optim.step()
            
            # Set 'requires_grad' of the discriminators as 'True'
            set_requires_grad([D_F, D_Q], True)

            # Calculate adversarial losses for the discriminators
            D_adv_loss_F = adv_loss(D_F(x_F), torch.ones_like(D_F(x_F))) + adv_loss(D_F(x_QF.detach()), torch.zeros_like(D_F(x_QF.detach())))
            D_adv_loss_Q = adv_loss(D_Q(x_Q), torch.ones_like(D_Q(x_Q))) + adv_loss(D_Q(x_FQ.detach()), torch.zeros_like(D_Q(x_FQ.detach())))
            D_total_loss_F = D_adv_loss_F / 2.0
            D_total_loss_Q = D_adv_loss_Q / 2.0

            # Update the discriminators
            D_optim.zero_grad()
            D_total_loss_F.backward()
            D_total_loss_Q.backward()
            D_optim.step()

            # Calculate the average loss during one epoch
            losses['G_adv_loss_F'](G_adv_loss_F.detach())
            losses['G_adv_loss_Q'](G_adv_loss_Q.detach())
            losses['G_cycle_loss_F'](G_cycle_loss_F.detach())
            losses['G_cycle_loss_Q'](G_cycle_loss_Q.detach())
            losses['G_iden_loss_F'](G_iden_loss_F.detach())
            losses['G_iden_loss_Q'](G_iden_loss_Q.detach())
            losses['D_adv_loss_F'](D_adv_loss_F.detach())
            losses['D_adv_loss_Q'](D_adv_loss_Q.detach())
    
        for name in loss_name:
            losses_list[name].append(losses[name].result())
        
        # Save the trained model and list of losses
        torch.save({'epoch': epoch + 1, 'G_F2Q_state_dict': G_F2Q.state_dict(), 'G_Q2F_state_dict': G_Q2F.state_dict(),
                        'D_F_state_dict': D_F.state_dict(), 'D_Q_state_dict': D_Q.state_dict(),
                        'G_optim_state_dict': G_optim.state_dict(), 'D_optim_state_dict': D_optim.state_dict()}, join(path_checkpoint, model_name + '.pth'))
        for name in loss_name:
            torch.save(losses_list[name], join(path_result, name + '.npy'))
            
    # Plot loss graph (adversarial loss)
    plt.figure(1)
    for name in ['G_adv_loss_F', 'G_adv_loss_Q', 'D_adv_loss_F', 'D_adv_loss_Q']:
        loss_arr = torch.load(join(path_result, name + '.npy'), map_location='cpu')
        x_axis = np.arange(1, len(loss_arr) + 1)
        plt.plot(x_axis, loss_arr, label=name)
        
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, 0.25))
    plt.legend(loc='upper right')
    plt.savefig(join(path_result, 'loss_curve_1.png'))
    plt.close() 
    
    # Plot loss graph (cycle consistency loss, identity loss)
    plt.figure(2)
    for name in ['G_cycle_loss_F', 'G_cycle_loss_Q', 'G_iden_loss_F', 'G_iden_loss_Q']:
        loss_arr = torch.load(join(path_result, name + '.npy'), map_location='cpu')
        plt.plot(x_axis, loss_arr, label=name)
        
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(join(path_result, 'loss_curve_2.png'))
    plt.close() 
    

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_checkpoint', type=str, default='./CT_denoising')
    parser.add_argument('--model_name', type=str, default='cyclegan_v1')
    parser.add_argument('--path_data', type=str, default='./AAPM_data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lambda_cycle', type=int, default=10)
    parser.add_argument('--lambda_iden', type=int, default=5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_epoch', type=int, default=120)
    parser.add_argument('--g_channels', type=int, default=32)
    parser.add_argument('--d_channels', type=int, default=64)
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num_res_blocks', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_checkpoint', action='store_true')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    train(
        path_checkpoint=args.path_checkpoint,
        model_name=args.model_name,
        path_data=args.path_data,
        batch_size=args.batch_size,
        lambda_cycle=args.lambda_cycle,
        lambda_iden=args.lambda_iden,
        beta1=args.beta1,
        beta2=args.beta2,
        num_epoch=args.num_epoch,
        g_channels=args.g_channels,
        d_channels=args.d_channels,
        ch_mult=args.ch_mult,
        num_res_blocks=args.num_res_blocks,
        lr=args.lr,
        use_checkpoint=args.use_checkpoint
    )