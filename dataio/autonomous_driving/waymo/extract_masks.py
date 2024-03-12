"""
@file   extract_masks.py
@brief  Extract semantic mask

Using SegFormer, 2021. Cityscapes 83.2%
Relies on timm==0.3.2 & pytorch 1.8.1 (buggy on pytorch >= 1.9)

Installation:
    NOTE: mmcv-full==1.2.7 requires another pytorch version & conda env.
        Currently mmcv-full==1.2.7 does not support pytorch>=1.9; 
            will raise AttributeError: 'super' object has no attribute '_specify_ddp_gpu_num'
        Hence, a seperate conda env is needed.

    git clone https://github.com/NVlabs/SegFormer

    conda create -n segformer python=3.8
    conda activate segformer
    # conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

    pip install timm==0.3.2 pylint debugpy opencv-python attrs ipython tqdm imageio scikit-image omegaconf
    pip install mmcv-full==1.2.7 --no-cache-dir
    
    cd SegFormer
    pip install .

Usage:
    Direct run this script in the newly set conda env.
"""


from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

if __name__ == "__main__":
    import os
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
    parser.add_argument('--mask_dirname', type=str, default="masks")

    # Algorithm configs
    parser.add_argument('--segformer_path', type=str, default='/home/guojianfei/ai_ws/SegFormer')
    parser.add_argument('--config', help='Config file', type=str, default=None)
    parser.add_argument('--checkpoint', help='Checkpoint file', type=str, default=None)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette', default='cityscapes', help='Color palette used for segmentation map')
    
    args = parser.parse_args()
    if args.config is None:
        args.config = os.path.join(args.segformer_path, 'local_configs', 'segformer', 'B5', 'segformer.b5.1024x1024.city.160k.py')
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.segformer_path, 'pretrained', 'segformer.b5.1024x1024.city.160k.pth')
    
    if args.seq_list is not None:
        with open(args.seq_list, 'r') as f:
            seq_list = f.read().splitlines()
        select_scene_ids = [s.split(',')[0].rstrip(".tfrecord") for s in seq_list]
    else:
        select_scene_ids = list(sorted(glob(os.path.join(args.data_root, "*", "scenario.pt"))))
        select_scene_ids = [os.path.split(os.path.dirname(s))[-1] for s in select_scene_ids]
    
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    
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
                    img = model.show_result(fpath, result, palette=get_palette(args.palette), show=False)
                    imageio.imwrite(os.path.join(mask_dir, f"{fbase}.jpg"), img)
                
                # tmp = (~(mask==10)).astype(np.float)
                
                # import matplotlib.pyplot as plt
                # plt.imshow(result[0])
                
                # show_result_pyplot(model, fpath, result, get_palette(args.palette))
                
                