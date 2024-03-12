"""
@file   gui_runner_forest.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Visualizer using neural rendering based on modified kaolin-wisp framework.
"""

if __name__ == "__main__":

    from nr3d_lib.gui.kaolin_wisp_modified.cuda_guard import setup_cuda_context
    setup_cuda_context()     # Must be called before any torch operations take place
    
    import os
    import functools
    import numpy as np
    
    import torch
    from torch.utils.benchmark import Timer

    from nr3d_lib.utils import import_str, IDListedDict
    from nr3d_lib.config import BaseConfig
    from nr3d_lib.checkpoint import sorted_ckpts

    from nr3d_lib.config import ConfigDict
    from nr3d_lib.maths import inverse_transform_matrix

    from nr3d_lib.models.accelerations import OccGridEmaBatched

    from app.resources import Scene, AssetBank, load_scene_bank, create_scene_bank
    from app.resources.observers import Camera
    
    bc = BaseConfig()
    bc.parser.add_argument("--class_name", type=str, default='Street', help="The class_name of the object you want to operate with.")
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--slice_at", type=int, default=0, help="Specifies the frame at which the scene is frozen in advance of rendering.")
    bc.parser.add_argument("--downscale", type=float, default=2., help="Sets the side length downscale for rendering and output.")
    args = bc.parse()
    
    device = torch.device('cuda')
    if (ckpt_file:=args.get('load_pt', None)) is None:
        # Automatically load 'final_xxx.pt' or 'latest.pt'
        ckpt_file = sorted_ckpts(os.path.join(args.exp_dir, 'ckpts'))[-1]
    state_dict = torch.load(ckpt_file, map_location=device)
    
    #---------------------------------------------
    #-----------     Scene Bank     --------------
    #---------------------------------------------
    scene_bank: IDListedDict[Scene] = IDListedDict()
    scenebank_root = os.path.join(args.exp_dir, 'scenarios')
    scene_bank, scenebank_meta = load_scene_bank(scenebank_root, device=device)
    
    #---------------------------------------------
    #-----------     Asset Bank     --------------
    #---------------------------------------------
    asset_bank = AssetBank(args.assetbank_cfg)
    asset_bank.create_asset_bank(scene_bank, scenebank_meta, load_state_dict=state_dict['asset_bank'], device=device)
    asset_bank.model_setup()
    asset_bank.eval()
    
    #---------------------------------------------
    #---     Load assets to scene objects     ----
    #---------------------------------------------
    # for scene in scene_bank:
    scene = scene_bank[0]
    scene.load_assets(asset_bank)
    # !!! Only call training_before_per_step when all assets are ready & loaded !
    asset_bank.training_before_per_step(args.training.num_iters) # NOTE: Finished training.

    scene.slice_at(args.slice_at)
    
    bg_obj = scene.get_drawable_groups_by_class_name(args.class_name)[0]
    bg_model = bg_obj.model
    bg_model.ray_query_cfg.forward_inv_s = 6400.
    bg_model.ray_query_cfg.query_param = ConfigDict(
        should_sample_coarse=False, 
        march_cfg = ConfigDict(
            step_size = 0.05, 
            max_steps = 4096,
        ), 
        num_fine = 4, 
        upsample_inv_s = 64., 
        upsample_inv_s_factors = [4, 16]
    )
    
    # resolution = int(bg_model.space.world_block_size[0].item() / 0.2)
    # resolution = int(bg_model.space.world_block_size[0].item() / 0.4)
    # print("resolution=", resolution)
    # bg_model.accel.occ = OccGridEmaBatched(
    #     bg_model.space.n_trees, resolution, occ_val_fn_cfg=ConfigDict(type='sdf', inv_s=256.0), occ_thre=0.8, 
    #     init_cfg=ConfigDict(num_steps=128, num_pts=2**24))
    # # bg_model.accel.occ._init_from_net( num_steps=64, num_pts_per_batch=2**20)
    # bg_model.training_before_per_step(0) # Run init_from_net
    bg_model.training_before_per_step(args.training.num_iters) # Run init_from_net

    # ray_query_cfg = ConfigDict(perturb=False, with_rgb=True, with_normal=False, nablas_has_grad=False)
    ray_query_cfg = ConfigDict(perturb=False, with_rgb=True, with_normal=True, nablas_has_grad=False)
    ray_query_cfg.update(bg_model.ray_query_cfg)

    cam0: Camera = scene.get_observer_groups_by_class_name('Camera', False)[0]
    
    
    # with torch.no_grad():
    #     cam0.intr.set_downscale(2)
    #     rays_o, rays_d = cam0.get_all_rays()
    #     (rays_o, *_), (rays_d, *_) = scene.convert_rays_in_nodes_list(rays_o, rays_d, [bg_obj])
    
    # @torch.no_grad()
    # def to_img(tensor: torch.Tensor, H=cam0.intr.H, W=cam0.intr.W):
    #     *_, num_samples, channels = tensor.shape
    #     assert num_samples == H * W
    #     return tensor.view([H, W, channels]).data.cpu().numpy()

    # @torch.no_grad()
    # def render_fn(rays_o, rays_d, config):
    #     ray_input=dict(rays_o=rays_o, rays_d=rays_d, near=0.1,  far=120.0)
    #     ray_tested = bg_model.space.ray_test(**ray_input)
    #     ret = bg_model.ray_query(ray_input=ray_input, ray_tested=ray_tested, config=config, render_per_obj_individual=True)
    #     return ret

    # ret = batchify_query(functools.partial(render_fn, config=ray_query_cfg), rays_o, rays_d, chunk=2**16, show_progress=True)
    
    """
        neuralsim / neuralgen / neurecon uses: < opencv / colmap convention >                 
        facing [+z] direction, y downwards, x right
                  z                                   
                 /                                     
                o------> x                             
                |                                      
                ↓ 
                y

        kaolin Camera (default): < opengl convention >
        facing [-z] direction, y upwards, x right
                y
                ^
                |
                o------> x
               /
             z
    """
    opengl2opencv = torch.tensor(
        [[1, 0, 0, 0], 
        [0, -1, 0, 0],
        [0, 0, -1, 0], 
        [0, 0, 0, 1]], dtype=torch.float, device=device)

    # camera opengl to world =  camera opencv to world @ opengl2opencv
    c2w_opengl = cam0.world_transform.mat_4x4() @ opengl2opencv
    w2c = inverse_transform_matrix(c2w_opengl)
    
    from kaolin.render.camera import Camera as klCamera
    from nr3d_lib.gui.kaolin_wisp_modified.wisp_app import APP
    from nr3d_lib.gui.neural_renderer import NR3DKaolinWispRenderer
    
    cam0.intr.set_downscale(4)
    kaolin_camera = klCamera.from_args(view_matrix=w2c, focal_x=cam0.intr.focal()[0].item(), width=cam0.intr.W, height=cam0.intr.H, near=0.1, far=200.)
    state=ConfigDict(
        renderer=ConfigDict(
            antialiasing='msaa_4x', 
            clear_color_value=(0,0,0),
            canvas_height=cam0.intr.H, 
            canvas_width=cam0.intr.W, 
            target_fps=24, 
            device=device, 
            selected_camera=kaolin_camera, 
            selected_camera_lens="perspective", 
            reference_grids=['xz']
        ), 
        nr_data_layers=ConfigDict()
    )
    neural_renderer = NR3DKaolinWispRenderer(bg_model, ray_query_cfg)
    app = APP("neuralsim v0.4.3", state, neural_renderer)
    app.run()