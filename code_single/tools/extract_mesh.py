"""
@file   extract_mesh.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Extract mesh from trained neural fields.
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

import torch

from nr3d_lib.fmt import log
from nr3d_lib.config import ConfigDict
from nr3d_lib.checkpoint import sorted_ckpts
from nr3d_lib.utils import IDListedDict, cond_mkdir, import_str
from nr3d_lib.models.spatial import ForestBlockSpace
from nr3d_lib.graphics.trianglemesh import extract_mesh

from dataio.scene_dataset import SceneDataset
from app.resources import Scene, load_scene_bank, AssetBank

def main_function(args: ConfigDict):
    #---------------------------------------------
    #--------------     Load     -----------------
    #---------------------------------------------
    device = torch.device('cuda')
    if (ckpt_file:=args.get('load_pt', None)) is None:
        # Automatically load 'final_xxx.pt' or 'latest.pt'
        ckpt_file = sorted_ckpts(os.path.join(args.exp_dir, 'ckpts'))[-1]
    log.info("=> Use ckpt:" + str(ckpt_file))
    state_dict = torch.load(ckpt_file, map_location=device)

    #---------------------------------------------
    #-----------     Scene Bank     --------------
    #---------------------------------------------
    scene_bank: IDListedDict[Scene] = IDListedDict()
    scenebank_root = os.path.join(args.exp_dir, 'scenarios')
    scene_bank, _ = load_scene_bank(scenebank_root, device=device)

    #---------------------------------------------
    #-----------     Asset Bank     --------------
    #---------------------------------------------
    asset_bank = AssetBank(args.assetbank_cfg)
    asset_bank.create_asset_bank(scene_bank, load_state_dict=state_dict['asset_bank'], device=device)
    # log.info(asset_bank)

    #---------------------------------------------
    #---     Load assets to scene objects     ----
    #---------------------------------------------
    # for scene in scene_bank:
    scene = scene_bank[0]
    scene.load_assets(asset_bank)
    # !!! Only call training_before_per_step when all assets are ready & loaded !
    asset_bank.training_before_per_step(args.training.num_iters) # NOTE: Finished training.

    dataset_impl: SceneDataset = import_str(args.dataset_cfg.target)(args.dataset_cfg.param)
    scale_mat = dataset_impl.scale_mat if hasattr(dataset_impl, "scale_mat") else None

    #---------------------------------------------
    #----------     Mesh Extraction     ----------
    #---------------------------------------------
    mesh_root = os.path.join(args.exp_dir, args.dirname)
    cond_mkdir(mesh_root)

    with torch.no_grad():
        if args.res is not None:
            log.info(f"Start [extract_mesh], res={args.res}, levelset={args.levelset}, in {args.exp_dir}")
        else:
            log.info(f"Start [extract_mesh], N={args.N}, levelset={args.levelset}, in {args.exp_dir}")
        for scene in scene_bank:
            scene.slice_at(args.slice_at)
            obj = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]
            
            if args.to_world:
                transform_mat = obj.world_transform.mat_4x4().data.cpu().numpy()
                transform_mat = transform_mat @ scale_mat if scale_mat is not None else transform_mat
            else:
                transform_mat = scale_mat
            
            model = obj.model
            aabb = model.space.aabb
            size = (aabb[1] - aabb[0])
            if args.res is not None:
                args.N = max(int(size.min().item() / args.res + 0.5), 1)

            mesh_file_base = f"{scene.id}#{obj.class_name}#{obj.id}_level={args.levelset}" + (f"_N={args.N}" if args.res is None else f"_res={args.res}")
            mesh_file_base = args.outbase or mesh_file_base
            mesh_file = os.path.join(mesh_root, f"{mesh_file_base}.ply")
            
            if scene.image_embeddings is not None:
                # Use the 0-th image embedding
                cam0_id = list(scene.observer_groups_by_class_name['Camera'].keys())[0]
                h_appear = scene.image_embeddings[cam0_id](torch.tensor([0], device=device))
            else:
                h_appear = None
            
            if args.surface_type == 'sdf':
                levelset = args.levelset
                query_surf_fn = lambda x: model.forward_sdf(x, input_normalized=False)['sdf']
                query_color_fn = lambda x, v: model.forward(
                    x, v, with_rgb=True, with_normal=False, input_normalized=False, 
                    h_appear=h_appear.expand(x.shape[0], -1) if h_appear is not None else None)['rgb']
            elif args.surface_type == 'nerf':
                # NOTE: *(-1) since larger density value means inside.
                levelset = -1 * args.levelset
                query_surf_fn = lambda x: -1 * model.forward_density(x, input_normalized=False)['sigma']
                query_color_fn = lambda x, v: model.forward(
                    x, v, input_normalized=False, 
                    h_appear=h_appear.expand(x.shape[0], -1) if h_appear is not None else None)['rgb']

            else:
                raise RuntimeError(f"Invalid surface_type={args.surface_type}")
            
            aabb = model.space.aabb.flatten().tolist()
            extract_mesh(query_surf_fn, query_color_fn, bmin=aabb[:3], bmax=aabb[3:], N=args.N,
                         include_color=args.include_color, filepath=mesh_file, show_progress=True,
                         chunk=args.chunk, scale=args.scale, transform=transform_mat, level=levelset)
            scene.unfrozen()

def make_parser():
    from nr3d_lib.config import BaseConfig
    bc = BaseConfig()
    
    bc.parser.add_argument('--N', type=int, default=512, 
                           help='Defines the resolution of the marching cube algorithm. "\
                           "\n If the space is cuboid-shaped, it represents the resolution of the shortest edge of the cuboid.')
    bc.parser.add_argument('--res', type=float, default=None, 
                           help='Sets the grid side length (in world) for the marching cube algorithm. "\
                           "\n If the space is cuboid-shaped, it represents the resolution of the shortest edge of the cuboid.')
    bc.parser.add_argument('--scale', type=float, default=None, help="Optionally sets a scaling factor to the output mesh.")

    bc.parser.add_argument('--surface_type', type=str, default='sdf', help="Options: [sdf, nerf]. Use `sdf` by default.")
    bc.parser.add_argument("--include_color", action='store_true', help="If set, the appearance color will be attached to the mesh.")
    bc.parser.add_argument("--to_world", action='store_true', help="If set, the mesh will be transformed to world coordinates.")
    bc.parser.add_argument("--slice_at", type=int, default=0, help="Specifies the frame at which the scene is frozen before extracting the mesh.")
    bc.parser.add_argument("--levelset", type=float, default=0.0, help="Defines the level set to extract the surface.")
    
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--chunk", type=int, default=16*1024, help='Chunkify the quering process. Modify for smaller GPU mem.')
    bc.parser.add_argument("--dirname", type=str, default="meshes", help="Sets the output directory to /path/to/exp_dir/${dirname}`.")
    bc.parser.add_argument("--outbase", type=str, default=None, help="Sets the basename of the output file (without extension).")
    return bc

if __name__ == "__main__":
    bc = make_parser()
    main_function(bc.parse())