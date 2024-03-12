import os
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim

from nr3d_lib.utils import load_rgb
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.utils import get_scheduler
from nr3d_lib.models.loss.recon import huber_loss
from nr3d_lib.models.grid_encodings.permuto import GenerativePermutoConcat

n_latent_dim = 1
num_iters = 10000
num_pixels = 100000
lr = 0.01
"""
0.03 ~ 0.1 barely allows the all-zero-initialized latent to exhibit a monotonically increasing behavior
Below 0.01 is still not quite enough
"""
wreg = 0.03
use_scheduler = True
min_factor = 0.2
# param_dtype = torch.float16
param_dtype = torch.float32
device = torch.device('cuda')

m = GenerativePermutoConcat(
    n_latent_dim, 2, 3, 
    encoding_cfg=ConfigDict(permuto_auto_compute_cfg=ConfigDict(
        type='multi_res', coarsest_res=4.0, finest_res=2000.0, n_levels=16, n_feats=4, 
        log2_hashmap_size=18, apply_random_shifts_per_level=True)), 
    decoder_cfg=ConfigDict(type='mlp', D=2, W=64, output_activation='sigmoid'), dtype=param_dtype, device=device
)

print(m)

data_root = '/data1/video_dataset/london_cut_0014/raw_images'
imgs = [load_rgb(p) for p in sorted(glob(os.path.join(data_root, '*.png')))]
imgs = [torch.tensor(im, dtype=torch.float) for im in imgs]

latents_init = torch.zeros([len(imgs), n_latent_dim], dtype=torch.float, device=device)
# latents_init = torch.linspace(0, 1, len(imgs), dtype=torch.float, device=device)
# latents_init = torch.zeros([len(imgs), n_latent_dim], dtype=torch.float, device=device).uniform_(-0.1, 0.1)
latents = nn.Embedding(len(imgs),  n_latent_dim, dtype=torch.float, device=device, _weight=latents_init.clone())

optimer = optim.Adam(list(latents.parameters())+list(m.parameters()), lr=lr, eps=1.0e-15, betas=(0.9, 0.99))
if use_scheduler:
    scheduler = get_scheduler(
        ConfigDict(type='exponential', min_factor=min_factor, warmup_steps=500, num_iters=num_iters, ), 
        optimizer=optimer)

logging_loss = []

with tqdm(range(num_iters)) as pbar:
    for _ in pbar:
        im_ind = np.random.randint(0, len(imgs))
        im = imgs[im_ind]
        H, W, _ = im.shape
        xy = torch.rand([num_pixels, 2], dtype=torch.float, device=device)
        wh = (xy * xy.new_tensor([W,H])).long()
        wh.clamp_(wh.new_tensor([0]), wh.new_tensor([W-1, H-1]))
        
        z = latents(torch.tensor([[im_ind]], dtype=torch.long, device=device))
        rgb_pred = m.forward(xy.unsqueeze(0), z=z/2+0.5)['output']
        rgb_gt = im[wh[:, 1], wh[:, 0]].to(device).unsqueeze(0)
        
        loss = huber_loss(rgb_pred, rgb_gt)
        # latent close by
        loss += wreg * latents.weight.diff(dim=0).norm(dim=-1).mean()
        
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        if use_scheduler:
            scheduler.step()
        
        logging_loss.append(loss.item())
        pbar.set_postfix(lr=optimer.param_groups[0]['lr'], loss=loss.item())

@torch.no_grad()
def pred_im(z: torch.Tensor, HW: Tuple[int,int]):
    z = z.view(-1, n_latent_dim)
    B = z.size(0)
    H, W = HW
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='xy')
    wh = torch.stack([i+0.5, j+0.5], dim=-1)
    xy = wh / wh.new_tensor([W,H])
    xy = xy.tile(B,1,1,1) # BHW2
    im = m.forward(xy, z=z/2+0.5)['output'] # BHW3
    return im

z_test = latents(torch.tensor([[0]], dtype=torch.long, device=device))
im_test = pred_im(z_test, (800, 1200))

plt.plot(logging_loss)
plt.show()

plt.imshow(im_test[0].data.float().cpu().numpy())
plt.show()

latents_new = latents.weight.data.clone()
z_vid = torch.linspace(-latents_new.min().item(),-latents_new.max().item(),24,dtype=torch.float, device=device).unsqueeze(-1).tile(1,n_latent_dim)
vid = pred_im(z_vid, (800, 1200)).data.float().cpu().numpy()
vid = (vid*255.).clip(0,255).astype(np.uint8)
vid_pth = 'dev_test/test_permuto_vid.mp4'
imageio.mimwrite(vid_pth, vid)
print(f"Video saved to {vid_pth}")
