"""
@file   extract_masks_vit_adapter.py
@brief  Extract semantic mask

NOTE: 
    Using ViT-Adapter, 2022. ADE20K
    In practice, the performance of sky segmentation of ViT-Adapter trained on cityscapes is very poor.
    Hence we use the its version that's trained on ADE20K instead.
    
    Relies on timm==0.4.12 & pytorch 1.9.0

Installation:
    NOTE: mmcv-full==1.4.2 requires another pytorch version.

    git clone https://github.com/czczup/ViT-Adapter
    
    conda create -n vitadapter python=3.8 
    conda activate vitadapter

    pip install imageio scipy tqdm
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
    pip install timm==0.4.12
    pip install mmdet==2.22.0 # for Mask2Former
    pip install mmsegmentation==0.20.2

    cd ViT-Adapter/segmentation
    ln -s ../detection/ops ./
    cd ops
    sh make.sh # compile deformable attention
"""

algo_root = '/home/guojianfei/ai_ws/ViT-Adapter/segmentation'
data_root = '/data1/waymo/processed'

import sys
sys.path.append(algo_root)

import torch

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes

if __name__ == "__main__":
    import os
    import sys
    import imageio
    import numpy as np
    from glob import glob
    from tqdm import tqdm
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Custom configs
    parser.add_argument('--data_root', type=str, default='/data1/waymo/processed')
    parser.add_argument('--seq_list', type=str, default=None, help='specify --seq_list if you want to limit the list of seqs')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--no_compress', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--mask_dirname', type=str, default="masks_vit_adapter")
    # Algorithm configs
    parser.add_argument(
        '--config', help='Config file', type=str, 
        # default=os.path.join(algo_root, 'configs', 'cityscapes', 'mask2former_beit_adapter_large_896_80k_cityscapes_ss.py'))
        # default=os.path.join(algo_root, 'configs', 'ade20k', 'upernet_beit_adapter_large_640_160k_ade20k_ss.py'))
        default=os.path.join(algo_root, 'configs', 'ade20k', 'mask2former_beitv2_adapter_large_896_80k_ade20k_ss.py'))
    parser.add_argument(
        '--checkpoint', help='Checkpoint file', type=str, 
        # default=os.path.join(algo_root, 'mask2former_beit_adapter_large_896_80k_cityscapes.pth.tar')) # Terrible sky
        # default=os.path.join(algo_root, 'upernet_beit_adapter_large_640_160k_ade20k.pth.tar')) # 1.2 fps
        default=os.path.join(algo_root, 'mask2former_beitv2_adapter_large_896_80k_ade20k.pth')) # 0.31 fps
    
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')

    args = parser.parse_args()
    if args.seq_list is not None:
        with open(args.seq_list, 'r') as f:
            seq_list = f.read().splitlines()
        select_scene_ids = [s.split(',')[0].rstrip(".tfrecord") for s in seq_list]
    else:
        select_scene_ids = list(sorted(glob(os.path.join(args.data_root, "*", "scenario.pt"))))
        select_scene_ids = [os.path.split(os.path.dirname(s))[-1] for s in select_scene_ids]

    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    for scene_i, scene_id in enumerate(tqdm(select_scene_ids, f'Extracting masks ...')):
        obs_id_list = sorted(os.listdir(os.path.join(args.data_root, scene_id, args.rgb_dirname)))
        
        for obs_i, obs_id in enumerate(tqdm(obs_id_list, f'scene [{scene_i}/{len(select_scene_ids)}]')):
            img_dir = os.path.join(args.data_root, scene_id, args.rgb_dirname, obs_id)
            mask_dir = os.path.join(args.data_root, scene_id, args.mask_dirname, obs_id)
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            
            flist = sorted(glob(os.path.join(img_dir, '*.jpg')))
            for fpath in tqdm(flist, f'scene[{scene_i}][{obs_id}]'):
                fbase = os.path.splitext(os.path.basename(os.path.normpath(fpath)))[0]
                if args.no_compress:
                    mask_fpath = os.path.join(mask_dir, f"{fbase}.npy")
                else:
                    mask_fpath = os.path.join(mask_dir, f"{fbase}.npz")
                
                if args.ignore_existing and os.path.exists(mask_fpath):
                    continue
                
                #---- Inference and save outputs
                result = inference_segmentor(model, fpath)
                mask = result[0].astype(np.uint8)   # NOTE: in the settings of "cityscapes", there are 19 classes at most.
                if args.no_compress:
                    np.save(mask_fpath, mask)
                else:
                    np.savez_compressed(mask_fpath, mask)   # NOTE: compressed files are 100x smaller.
                
                if args.verbose:
                    if hasattr(model, 'module'):
                        model = model.module
                    img = model.show_result(fpath, result, palette=get_palette(args.palette), show=False, opacity=0.5)
                    imageio.imwrite(os.path.join(mask_dir, f"{fbase}.jpg"), img)
                
                # tmp = (~(mask==10)).astype(np.float)
                
                # import matplotlib.pyplot as plt
                # plt.imshow(result[0])
                
                # show_result_pyplot(model, fpath, result, get_palette(args.palette))