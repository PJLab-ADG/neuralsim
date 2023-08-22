"""
@file   extract_mono_cues.py
@brief  extract monocular cues (normal & depth)
        Adapted from https://github.com/EPFL-VILAB/omnidata

installation:
    pip install einops joblib pandas h5py scipy seaborn kornia timm pytorch-lightning
"""

import os
import sys
import argparse
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
from typing import Literal

import PIL
import skimage
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as transF


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img

def save_outputs(img_path: str, output_path_base: str, *, ref_img_size: int=384, mask_path: str = None, verbose=True):
    from dataio.dtu.dtu_dataset import load_mask
    with torch.no_grad():
        
        # img = Image.open(img_path)
        img = imageio.imread(img_path)
        img = skimage.img_as_float32(img)
        
        if mask_path is not None:
            mask = load_mask(mask_path)
            img[mask] = 0.
        
        H, W, _ = img.shape
        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)
        if H < W:
            H_ = ref_img_size
            W_ = int(((W / (H/H_)) // 32)) * 32 # Force to be a multiple of 32
        else:
            W_ = ref_img_size
            H_ = int(((H / (W/W_)) // 32)) * 32 # Force to be a multiple of 32
        img_tensor = transF.resize(img_tensor, (H_, W_), antialias=True)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)

        if args.task == 'depth':
            #output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
            output = output.clamp(0,1).data / output.max()
            # output = transF.resize(output, (H,W), antialias=True).movedim(0,-1).mul_(255.).to(torch.uint8).clamp_(0,255).cpu().numpy()
            output = transF.resize(output, (H,W), antialias=True).movedim(0,-1).cpu().numpy()
            # output = transF.resize(output, (H,W), antialias=True).movedim(0,-1).cpu().numpy()
            
            # np.savez_compressed(f"{output_path_base}.npz", output)
            # output = 1 - output
            # output = standardize_depth_map(output)
            # plt.imsave(f"{output_path_base}.png", output, cmap='viridis')
            if verbose:
                imageio.imwrite(f"{output_path_base}.jpg", (output*255).clip(0,255).astype(np.uint8)) # Fastest and smallest file size
            # NOTE: jianfei: Although saving to float16 is lossy, we are allowing it since it's just extracting some weak hint here.
            np.savez_compressed(f"{output_path_base}.npz", output.astype(np.float16))
            
        else:
            output = output.data.clamp(0,1).squeeze(0)
            # Resize to original shape
            # NOTE: jianfei: Although saving to uint8 is lossy, we are allowing it since it's just extracting some weak hint here.
            output = transF.resize(output, (H,W), antialias=True).movedim(0,-1).mul_(255.).to(torch.uint8).clamp_(0,255).cpu().numpy()
            
            # np.savez_compressed(f"{output_path_base}.npz", output)
            # plt.imsave(f"{output_path_base}.png", output/2+0.5)
            imageio.imwrite(f"{output_path_base}.jpg", output) # Fastest and smallest file size
            # imageio.imwrite(f"{output_path_base}.png", output) # Very slow
            # np.save(f"{output_path_base}.npy", output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

    # Dataset specific configs
    parser.add_argument('--data_root', type=str, default='/data1/neus')
    parser.add_argument('--with_mask', action='store_true', help='estimate depth and normal with GT mask input')
    parser.add_argument('--no_verbose', action='store_true')

    # Original configs
    parser.add_argument('--omnidata_path', dest='omnidata_path', help="path to omnidata model")
    parser.set_defaults(omnidata_path='/home/guojianfei/ai_ws/omnidata/omnidata_tools/torch/')
    parser.add_argument('--pretrained_models', dest='pretrained_models', help="path to pretrained models")
    parser.set_defaults(pretrained_models='/home/guojianfei/ai_ws/omnidata/omnidata_tools/torch/pretrained_models/')
    parser.add_argument('--task', dest='task', help="normal or depth")
    parser.set_defaults(task='NONE')
    parser.add_argument('--ref_img_size', dest='ref_img_size', type=int, help="image size when inference (will still save full-scale output)")
    parser.set_defaults(ref_img_size=512)
    args = parser.parse_args()
    
    #-----------------------------------------------
    #-- Original preparation
    #-----------------------------------------------
    root_dir = args.pretrained_models 
    omnidata_path = args.omnidata_path

    sys.path.append(args.omnidata_path)
    print(sys.path)
    from modules.unet import UNet
    from modules.midas.dpt_depth import DPTDepthModel
    from data.transforms import get_transform

    trans_topil = transforms.ToPILImage()
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # get target task and model
    if args.task == 'normal':
        image_size = 384
        
        ## Version 1 model
        # pretrained_weights_path = root_dir + 'omnidata_unet_normal_v1.pth'
        # model = UNet(in_channels=3, out_channels=3)
        # checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

        # if 'state_dict' in checkpoint:
        #     state_dict = {}
        #     for k, v in checkpoint['state_dict'].items():
        #         state_dict[k.replace('model.', '')] = v
        # else:
        #     state_dict = checkpoint
        
        
        pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        # trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
        #                                     transforms.CenterCrop(image_size),
        #                                     get_transform('rgb', image_size=None)])

        trans_totensor = transforms.Compose([
            get_transform('rgb', image_size=None)])

    elif args.task == 'depth':
        image_size = 384
        pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
        # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
        model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        # trans_totensor = transforms.Compose([transforms.Resize(args.ref_img_size, interpolation=PIL.Image.BILINEAR),
        #                                     transforms.CenterCrop(image_size),
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize(mean=0.5, std=0.5)])
        trans_totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)])

    else:
        print("task should be one of the following: normal, depth")
        sys.exit()
    
    import shutil
    
    #-----------------------------------------------
    #--- Dataset Specific processing
    #-----------------------------------------------
    dtu_scan_dirs = list(sorted(os.listdir(args.data_root)))
    for scan_dirname in tqdm(dtu_scan_dirs, 'processsing scans...'):
        scan_dir = os.path.join(args.data_root, scan_dirname)
        img_dir = os.path.join(scan_dir, 'image')
        assert os.path.exists(img_dir)
        mask_dir = os.path.join(scan_dir, 'mask')
        if args.with_mask:
            assert os.path.exists(mask_dir)
        
        our_dir = os.path.join(scan_dir, args.task if not args.with_mask else f'{args.task}_wmask')
        os.makedirs(our_dir, exist_ok=True)
        
        img_names = list(sorted(os.listdir(img_dir)))
        for img_n in tqdm(img_names, scan_dirname):
            img_basename = os.path.splitext(img_n)[0]
            img_path = os.path.join(img_dir, img_n)
            img_ind = int(img_basename)
            mask_path = os.path.join(mask_dir, f"{img_ind:03d}.png")
            outpath_base = os.path.join(our_dir, img_basename)
            save_outputs(img_path, outpath_base, mask_path=(mask_path if args.with_mask else None), verbose=not args.no_verbose)