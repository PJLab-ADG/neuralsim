"""
@file   extract_mesh.py
@author Jianfei Guo, Shanghai AI Lab & Nianchen Deng, Shanghai AI Lab
@brief  Extract meshes of objects in a trained scene.
"""
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
set_env(2)

import os
import torch
import numpy as np
from omegaconf import OmegaConf

from nr3d_lib.fmt import log
from nr3d_lib.config import BaseConfig
from nr3d_lib.graphics.trianglemesh import extract_mesh
from nr3d_lib.models.spatial import AABBSpace, ForestBlockSpace, BatchedBlockSpace

from app.resources import load_scenes_and_assets
from app.visible_grid import VisibleGrid
from app.models.asset_base import AssetAssignment

def main(args, device=torch.device('cuda')):
    scene_bank, *_ = load_scenes_and_assets(**args, device=device)
    output_root = os.path.join(args.exp_dir, args.dirname)
    os.makedirs(output_root, exist_ok=True)

    #---------------------------------------------
    #----------     Mesh Extraction     ----------
    #---------------------------------------------
    reconstruct_cfg = OmegaConf.load(args.reconstruct_cfg)

    log.info(f"Start [extract_mesh] in {args.exp_dir}")
    with torch.no_grad():
        for scene in scene_bank:
            scene.frozen_at_global_frame(0)
            for class_name, class_cfg in reconstruct_cfg.items():
                if args.extract_classes and class_name not in args.extract_classes:
                    continue
                for i, obj in enumerate(scene.all_nodes_by_class_name[class_name]):
                    model = obj.model
                    if model.assigned_to in [AssetAssignment.MULTI_OBJ, AssetAssignment.MULTI_OBJ_ONE_SCENE]:
                        model.set_condition({"keys": [obj.full_unique_id]})

                    if "visible_grid" in class_cfg:
                        visible_grid = VisibleGrid.load(class_cfg.visible_grid, model.space)
                        visible_grid.build_accel()
                        visible_grid.postprocess("close2")
                        model.accel = visible_grid.accel
                    
                    mesh_file = os.path.join(output_root, f"{class_name}#{i}#{obj.id}_N={class_cfg['resolution']}.ply")
                    if isinstance(model.space, (ForestBlockSpace, AABBSpace)):
                        diameter3d = np.ones((3,))
                        aabb = class_cfg.get("aabb") or model.space.aabb.flatten().tolist()
                        query_sdf_fn = lambda x: model.forward_sdf(x, input_normalized=False)['sdf']
                        query_color_fn = lambda x, v: model.forward(x, v, with_rgb=True, with_normal=False, input_normalized=False)['rgb']
                    elif isinstance(model.space, BatchedBlockSpace):
                        diameter3d = model.space.radius3d * 2
                        aabb = class_cfg.get("aabb") or [-1., -1., -1., 1., 1., 1.]
                        query_sdf_fn = lambda x: model.forward_sdf(x, bidx=torch.zeros_like(x[..., 0]), input_normalized=True)['sdf']
                        query_color_fn = lambda x, v: model.forward(x, v, bidx=torch.zeros_like(x[..., 0]),
                                                                    input_normalized=True, with_rgb=True, with_normal=False)['rgb']
                    extract_mesh(query_sdf_fn, query_color_fn, bmin=aabb[:3], bmax=aabb[3:],
                                 N=class_cfg["resolution"], include_color=True, filepath=mesh_file,
                                 show_progress=True, chunk=args.chunk, scale=diameter3d)
            scene.unfrozen()

            
if __name__ == "__main__":
    bc = BaseConfig()
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--chunk", type=int, default=16*1024,
                           help='net chunk when querying the network. change for smaller GPU memory.')
    bc.parser.add_argument("--dirname", type=str, default='meshes', 
                           help="Sets the output directory to /path/to/exp_dir/${dirname}`.")
    bc.parser.add_argument("--reconstruct_cfg", type=str, default="reconstruct.yaml",
                           help="Path to the reconstruct configuration file. Defaults to \"reconstruct.yaml\" under `resume_dir`")
    bc.parser.add_argument("--extract_classes", type=str, nargs="+",
                           help="Only extract meshes of specified classes. Defaults to extract all classes defined in reconstruct configuration.")
    main(bc.parse())