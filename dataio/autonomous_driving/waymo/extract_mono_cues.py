"""
@file   extract_mono_cues.py
@brief  extract monocular cues (normal & depth)
        Adapted from https://github.com/EPFL-VILAB/omnidata

Installation:
    git clone https://github.com/EPFL-VILAB/omnidata

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

def list_contains(l: list, v):
    """
    Whether any item in `l` contains `v`
    """
    for item in l:
        if v in item:
            return True
    else:
        return False

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

def extract_cues(img_path: str, output_path_base: str, ref_img_size: int=384, verbose=True):
    with torch.no_grad():
        # img = Image.open(img_path)
        img = imageio.imread(img_path)
        img = skimage.img_as_float32(img)
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
            #output = 1 - output
#             output = standardize_depth_map(output)
            # plt.imsave(f"{output_path_base}.png", output, cmap='viridis')
            if verbose:
                imageio.imwrite(f"{output_path_base}.jpg", (output*255).clip(0,255).astype(np.uint8)[..., 0], format="jpg") # Fastest and smallest file size
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
    parser.add_argument('--data_root', type=str, default='/data1/waymo/processed')
    parser.add_argument('--seq_list', type=str, default=None, help='specify --seq_list if you want to limit the list of seqs')
    parser.add_argument('--verbose', action='store_true', help="Additionally generate .jpg files for visualization")
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--depth_dirname', type=str, default="depths")
    parser.add_argument('--normals_dirname', type=str, default="normals")

    # Algorithm configs
    parser.add_argument('--task', dest='task', required=True, default=None, help="normal or depth")
    parser.add_argument('--omnidata_path', dest='omnidata_path', help="path to omnidata model", 
                        default='/home/guojianfei/ai_ws/omnidata/omnidata_tools/torch/')
    parser.add_argument('--pretrained_models', dest='pretrained_models', help="path to pretrained models", 
                        default=None)
    parser.add_argument('--ref_img_size', dest='ref_img_size', type=int, default=512, 
                        help="image size when inference (will still save full-scale output)")
    args = parser.parse_args()

    #-----------------------------------------------
    #-- Original preparation
    #-----------------------------------------------
    if args.pretrained_models is None:
        # '/home/guojianfei/ai_ws/omnidata/omnidata_tools/torch/pretrained_models/'
        args.pretrained_models = os.path.join(args.omnidata_path, "pretrained_models")
    
    sys.path.append(args.omnidata_path)
    print(sys.path)
    from modules.unet import UNet
    from modules.midas.dpt_depth import DPTDepthModel
    from data.transforms import get_transform

    trans_topil = transforms.ToPILImage()
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get target task and model
    if args.task == 'normal' or args.task == 'normals':
        image_size = 384
        
        #---- Version 1 model
        # pretrained_weights_path = os.path.join(args.pretrained_models, 'omnidata_unet_normal_v1.pth')
        # model = UNet(in_channels=3, out_channels=3)
        # checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

        # if 'state_dict' in checkpoint:
        #     state_dict = {}
        #     for k, v in checkpoint['state_dict'].items():
        #         state_dict[k.replace('model.', '')] = v
        # else:
        #     state_dict = checkpoint
        
        
        pretrained_weights_path = os.path.join(args.pretrained_models, 'omnidata_dpt_normal_v2.ckpt')
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
        pretrained_weights_path = os.path.join(args.pretrained_models, 'omnidata_dpt_depth_v2.ckpt')  # 'omnidata_dpt_depth_v1.ckpt'
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

    #-----------------------------------------------
    #--- Dataset Specific processing
    #-----------------------------------------------
    if args.seq_list is not None:
        with open(args.seq_list, 'r') as f:
            seq_list = f.read().splitlines()
        select_scene_ids = [s.split(',')[0].rstrip(".tfrecord") for s in seq_list]
    else:
        select_scene_ids = list(sorted(glob(os.path.join(args.data_root, "*", "scenario.pt"))))
        select_scene_ids = [os.path.split(os.path.dirname(s))[-1] for s in select_scene_ids]
    
    for scene_i, scene_id in enumerate(tqdm(select_scene_ids, f'Extracting {args.task} ...')):
        obs_id_list = sorted(os.listdir(os.path.join(args.data_root, scene_id, args.rgb_dirname)))
        
        for obs_i, obs_id in enumerate(tqdm(obs_id_list, f'scene [{scene_i}/{len(select_scene_ids)}]')):
            img_dir = os.path.join(args.data_root, scene_id, args.rgb_dirname, obs_id)
            
            if args.task == 'depth':
                output_dir = os.path.join(args.data_root, scene_id, args.depth_dirname, obs_id)
            elif args.task == 'normal':
                output_dir = os.path.join(args.data_root, scene_id, args.normals_dirname, obs_id)
            else:
                raise RuntimeError(f"Invalid task={args.task}")
            os.makedirs(output_dir, exist_ok=True)
            
            flist = sorted(glob(os.path.join(img_dir, '*.jpg')))
            for fpath in tqdm(flist, f'scene[{scene_i}][{obs_id}]'):
                fbase = os.path.splitext(os.path.basename(os.path.normpath(fpath)))[0]
                
                if args.task == 'depth':
                    output_base = os.path.join(output_dir, fbase)
                elif args.task == 'normal':
                    output_base = os.path.join(output_dir, fbase)
                else:
                    raise RuntimeError(f"Invalid task={args.task}")
                
                if args.ignore_existing and list_contains(os.listdir(output_dir), fbase):
                    continue
                
                #---- Inference and save outputs
                extract_cues(fpath, output_base, args.ref_img_size, verbose=args.verbose)