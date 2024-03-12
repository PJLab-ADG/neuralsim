import os
import sys
def set_env(depth: int):
    # Add project root to sys.path
    current_file_path = os.path.abspath(__file__)
    project_root_path = os.path.dirname(current_file_path)
    for _ in range(depth):
        project_root_path = os.path.dirname(project_root_path)
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)
        print(f"Added {project_root_path} to sys.path")
set_env(1)

import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam
import torch.nn.functional as F

from nr3d_lib.graphics.cameras import pinhole_get_rays
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import check_to_torch, load_rgb
from nr3d_lib.models.utils import batchify_query, get_scheduler
from nr3d_lib.models.loss.recon import mse_loss

from app.models.env.neural_sky import SimpleSky

NUM_ITERS = 30000
LR = 1.0e-4

#---------------- Cityscapes semantic segmentation
cityscapes_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]
cityscapes_classes_ind_map = {cn: i for i, cn in enumerate(cityscapes_classes)}

device = torch.device('cuda')

img = load_rgb('/data1/waymo/processed/images/segment-7670103006580549715_360_000_380_000_with_camera_labels/camera_FRONT/00000000.jpg')
mask_all = np.load('/data1/waymo/processed/masks/segment-7670103006580549715_360_000_380_000_with_camera_labels/camera_FRONT/00000000.npz')['arr_0']
sky_mask = (mask_all==cityscapes_classes_ind_map['sky'])
sky_mask = check_to_torch(sky_mask, device=device).flatten(0)

H, W, *_ = img.shape
rgb_gt = check_to_torch(img, device=device, dtype=torch.float).flatten(0,-2)

# sky = SimpleSky({'type':'spherical', 'degree':8}, D=2, W=256, dtype=torch.float, device=device)
sky = SimpleSky({'type':'sinusoidal', 'n_frequencies':10}, D=2, W=256, dtype=torch.float, device=device)
optimzer = Adam(sky.parameters(), lr=LR)
scheduler = get_scheduler(ConfigDict(type='exponential', num_iters=NUM_ITERS, min_factor=0.03), optimzer)

with open('/data1/waymo/processed/scenarios/segment-7670103006580549715_360_000_380_000_with_camera_labels.pt', 'rb') as f:
    scenario = pickle.load(f)

obs_data = scenario['observers']['camera_FRONT']['data']
intr = check_to_torch(obs_data['intr'], device=device, dtype=torch.float)[0]
c2w = check_to_torch(obs_data['c2w'], device=device, dtype=torch.float)[0]

rays_o_all, rays_d_all = pinhole_get_rays(c2w, intr, H, W)

with tqdm(range(NUM_ITERS)) as pbar:
    for _ in pbar:
        optimzer.zero_grad()
        inds = torch.randint(H*W, [4096], device=device)
        rays_d = rays_d_all[inds]
        
        pred = sky.forward(F.normalize(rays_d, dim=-1))
        gt = rgb_gt[inds]
        
        loss = mse_loss(pred, gt, sky_mask[inds].unsqueeze(-1))
        loss.backward()
        
        pbar.set_postfix(loss=loss.item())
        
        optimzer.step()
        scheduler.step()

pred_img: torch.Tensor = batchify_query(sky.forward, rays_d_all, chunk=65536, show_progress=True)
pred_img = pred_img.reshape(H, W, 3).data.cpu().numpy()