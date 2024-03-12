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

import os
import numpy as np
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from nr3d_lib.config import ConfigDict

from nr3d_lib.models.blocks import LipshitzMLP, MLP
from nr3d_lib.models.fields_conditional.sdf.style_lotd_sdf import StyleLoTDSDF
from nr3d_lib.models.grid_encodings.lotd import TriplaneLoTDGrowerFlatten
from nr3d_lib.graphics.trianglemesh import extract_mesh
from nr3d_lib.models.grid_encodings.lotd.lotd_batched import LoTDBatched
from nr3d_lib.utils import cond_mkdir

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1.0e-4)
# NOTE: Best for now: 3k -> w=3.0e-5; 30k -> w=1.0e-6
parser.add_argument("--lipshitz", action='store_true')
parser.add_argument("--w_lip", type=float, default=3.0e-6)
parser.add_argument("--num_iters", type=int, default=3000)
args = parser.parse_args()

device = torch.device('cuda')
dtype  = torch.float32

n_latent_dim = 8
n_input_dim = 3

exp_dir = "./dev_test/lip/3d_cond/"
cond_mkdir(exp_dir)

if args.lipshitz:   
    m = StyleLoTDSDF(
        lotd_grower_cfg=ConfigDict(
            target="nr3d_lib.models.grid_encodings.lotd.TriplaneLoTDGrowerFlatten", 
            param=ConfigDict(
                z_dim=n_latent_dim, lod_res=[16,32,64], 
                lod_n_feats=4, D=4, W=256, use_lipshitz=True, c_init_factor=1.0
            )
        ), 
        decoder_cfg=ConfigDict(
            D=1, W=64, use_lipshitz=True, c_init_factor=1.0
        ), 
        device=device, dtype=dtype
    )
    
else:
    m = StyleLoTDSDF(
        lotd_grower_cfg=ConfigDict(
            target="nr3d_lib.models.grid_encodings.lotd.TriplaneLoTDGrowerFlatten", 
            param=ConfigDict(
                z_dim=n_latent_dim, lod_res=[16,32,64], 
                lod_n_feats=4, D=4, W=256, # use_lipshitz=True, c_init_factor=1.0
            )
        ), 
        decoder_cfg=ConfigDict(
            D=1, W=64, # use_lipshitz=True, c_init_factor=1.0
        ), 
        device=device, dtype=dtype
    )
print(m)

# https://iquilezles.org/articles/distfunctions/
def sdf_gt_cube(x: torch.Tensor, r=0.5):
    q = x.abs() - r
    return q.clamp_min(0.).norm(dim=-1) + q.max(dim=-1).values.clamp_max(0.)

def sdf_gt_cube_frame(x: torch.Tensor, r=0.5, e=0.1):
    p = x.abs() - r
    q = (p+e).abs() - e
    v1 = torch.stack([p[..., 0], q[..., 1], q[..., 2]], dim=-1)
    v2 = torch.stack([q[..., 0], p[..., 1], q[..., 2]], dim=-1)
    v3 = torch.stack([q[..., 0], q[..., 1], p[..., 2]], dim=-1)
    vs = torch.stack([v1, v2, v3], dim=0)
    return (vs.clamp_min(0.).norm(dim=-1) + vs.max(dim=-1).values.clamp_max(0.)).min(dim=0).values

def sdf_gt_torus(x: torch.Tensor, r1=0.6, r2=0.3):
    q = torch.stack([x[..., [0,2]].norm(dim=-1)-r1, x[..., 1]], dim=-1)
    return q.norm(dim=-1) - r2

def sdf_gt_sphere(x: torch.Tensor, r=0.5):
    return x.norm(dim=-1) - r

sdf_gt_fn1 = partial(sdf_gt_torus, r1=0.6, r2=0.3)
sdf_gt_fn2 = partial(sdf_gt_cube_frame, r=0.5)  
sdf_gt_fn3 = partial(sdf_gt_sphere, r=0.8)

extract_mesh(sdf_gt_fn1, None, N=128, filepath=os.path.join(exp_dir, 'gt1.ply'))
extract_mesh(sdf_gt_fn2, None, N=128, filepath=os.path.join(exp_dir, 'gt2.ply'))
extract_mesh(sdf_gt_fn3, None, N=128, filepath=os.path.join(exp_dir, 'gt3.ply'))

l1 = nn.Parameter(torch.empty([n_latent_dim], device=device).normal_(std=1.), requires_grad=True)
l2 = nn.Parameter(torch.empty([n_latent_dim], device=device).normal_(std=1.), requires_grad=True)
l3 = nn.Parameter(torch.empty([n_latent_dim], device=device).normal_(std=1.), requires_grad=True)

def sdf_pred_fn(x: torch.Tensor, l: torch.Tensor):
    m.set_condition(l)
    return m.forward(x.unsqueeze(0))['sdf'].squeeze(0)

optimzer = Adam(list(m.parameters())+[l1,l2,l3], lr=1.0e-3)

with tqdm(range(args.num_iters)) as pbar:
    for _ in pbar:
        optimzer.zero_grad()
        
        # [-1, 1]
        x = torch.rand([5000, n_input_dim], device=device) * 2 - 1
        
        sdf_pred1 = sdf_pred_fn(x, l1)
        sdf_pred2 = sdf_pred_fn(x, l2)
        sdf_pred3 = sdf_pred_fn(x, l3)
        
        sdf_gt1 = sdf_gt_fn1(x)
        sdf_gt2 = sdf_gt_fn2(x)
        sdf_gt3 = sdf_gt_fn3(x)
        
        loss = F.l1_loss(sdf_pred1, sdf_gt1) \
            + F.l1_loss(sdf_pred2, sdf_gt2) \
            + F.l1_loss(sdf_pred3, sdf_gt3)
            
        loss = loss.mean()
        
        if args.lipshitz:
            loss += args.w_lip * m.encoding.lotd_grower.mapper.lipshitz_bound_full()
            loss += args.w_lip * m.decoder.lipshitz_bound_full()
        
        loss.backward()
        optimzer.step()
        
        pbar.set_postfix(loss=loss.item())

with torch.no_grad():
    extract_mesh(lambda x: sdf_pred_fn(x, l1), None, N=128, filepath=os.path.join(exp_dir, 'pred1.ply'))
    extract_mesh(lambda x: sdf_pred_fn(x, l2), None, N=128, filepath=os.path.join(exp_dir, 'pred2.ply'))
    extract_mesh(lambda x: sdf_pred_fn(x, l2), None, N=128, filepath=os.path.join(exp_dir, 'pred3.ply'))
    
    alphas = np.linspace(0, 1, endpoint=True, num=10).tolist()
    mesh_files = []
    for alpha in alphas:
        l = l1 * (1-alpha) + l2 * alpha
        mesh_file = os.path.join(exp_dir, f'alpha={alpha:.2f}.ply')
        try:
            extract_mesh(lambda x: sdf_pred_fn(x, l), None, N=128, filepath=mesh_file)
            mesh_files.append(mesh_file)
        except:
            mesh_files.append(None)
    
    import open3d as o3d
    things_to_draw = []
    for i, (alpha, mesh_file) in enumerate(zip(alphas, mesh_files)):
        if mesh_file is None:
            continue
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh.compute_vertex_normals()
        mesh.translate([-2*i, 0, 0])
        things_to_draw.append(mesh)
    o3d.visualization.draw_geometries(things_to_draw)