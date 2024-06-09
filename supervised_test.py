import torch
import argparse
import numpy as np
from os import makedirs
from os.path import join, isdir
from tqdm.auto import tqdm
from cycleGAN_train import Generator
from supervised_train import make_dataloader
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


# Test function
def test(
    path_checkpoint = './CT_denoising',
    model_name = 'supervised_v1',
    path_data = '../data/AAPM_data',
    g_channels=32,
    ch_mult=[1, 2, 4, 8],
    num_res_blocks=3,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Path for saving checkpoint
    if not isdir(path_checkpoint):
        makedirs(path_checkpoint)

    # Path for saving results
    path_result = join(path_checkpoint, model_name)
    if not isdir(path_result):
        makedirs(path_result)
      
    test_dataloader = make_dataloader(path_data, is_train=False)
    
    # Load the last checkpoint
    G_Q2F = Generator(1, 1, g_channels, ch_mult=ch_mult, num_res_blocks=num_res_blocks).to(device)
    checkpoint = torch.load(join(path_checkpoint, model_name + '.pth'))
    G_Q2F.load_state_dict(checkpoint['G_Q2F_state_dict'])
    G_Q2F.eval()

    # Test and save
    with torch.no_grad():
        for _, x_Q, file_name in tqdm(test_dataloader):
            x_Q = x_Q.to(device)
            x_QF = G_Q2F(x_Q)[0].detach().cpu().numpy()
            x_QF = x_QF * 4000

            np.save(join(path_result, file_name[0]), x_QF[0])
    
    # Initialize lists for PSNR and SSIM
    psnr_quarter = []
    ssim_quarter = []
    psnr_output = []
    ssim_output = []

    # Calculate PSNR and SSIM for each test data
    for index in range (1, 421 + 1):
        path_quarter = join(path_data, f'test/quarter_dose/{index}.npy')
        path_full = join(path_data, f'test/full_dose/{index}.npy')
        path_output = join(path_result, f'{index}.npy')

        quarter = np.load(path_quarter)
        full = np.load(path_full)
        output = np.load(path_output)

        quarter = (quarter - 0.0192) / 0.0192 * 1000
        full = (full - 0.0192) / 0.0192 * 1000

        psnr_quarter.append(psnr(quarter, full))
        ssim_quarter.append(ssim(quarter, full))
        psnr_output.append(psnr(output, full))
        ssim_output.append(ssim(output, full))

    print('PSNR and SSIM')
    print('Mean PSNR between input and ground truth:')
    print(np.mean(psnr_quarter))
    print('Mean SSIM between input and ground truth:')
    print(np.mean(ssim_quarter))
    print('Mean PSNR between network output and ground truth:')
    print(np.mean(psnr_output))
    print('Mean SSIM between network output and ground truth:')
    print(np.mean(ssim_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_checkpoint', type=str, default='./CT_denoising')
    parser.add_argument('--model_name', type=str, default='supervised_v1')
    parser.add_argument('--path_data', type=str, default='./AAPM_data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--g_channels', type=int, default=32)
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num_res_blocks', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    test(
        path_checkpoint=args.path_checkpoint,
        model_name=args.model_name,
        path_data=args.path_data,
        g_channels=args.g_channels,
        ch_mult=args.ch_mult,
        num_res_blocks=args.num_res_blocks,
    )
