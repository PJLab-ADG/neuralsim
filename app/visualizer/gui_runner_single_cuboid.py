"""
@file   gui_runner_single_cuboid.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Visualizer using neural rendering based on modified kaolin-wisp framework.
"""

if __name__ == "__main__":
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

    from nr3d_lib.gui.kaolin_wisp_modified.cuda_guard import setup_cuda_context
    setup_cuda_context()     # Must be called before any torch operations take place

    import os
    import functools
    import numpy as np
    from icecream import ic

    import torch
    from torch.utils.benchmark import Timer

    from nr3d_lib.utils import import_str, IDListedDict
    from nr3d_lib.config import BaseConfig
    from nr3d_lib.checkpoint import sorted_ckpts

    from nr3d_lib.config import ConfigDict
    from nr3d_lib.maths import inverse_transform_matrix

    from nr3d_lib.models.accelerations import OccGridEma

    from app.resources import Scene, AssetBank, load_scene_bank, load_scenes_and_assets
    from app.resources.observers import Camera

    from nr3d_lib.models.grid_encodings.lotd import lotd
    lotd.hash_only = True

    bc = BaseConfig()
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"
                           "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--slice_at", type=int, default=0,
                           help="Specifies the frame at which the scene is frozen in advance of rendering.")
    bc.parser.add_argument("--downscale", type=float,
                           help="Sets the side length downscale for rendering and output.")
    bc.parser.add_argument("--regenerate_occgrid", type=float)
    args = bc.parse()

    device = torch.device("cuda")
    scene_bank, asset_bank, _ = load_scenes_and_assets(**args, device=device)
    scene = scene_bank[0]
    scene.slice_at(args.slice_at)

    obj = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]
    model = obj.model

    # Volume rendering config
    model.ray_query_cfg.query_param = ConfigDict({
        **model.ray_query_cfg.query_param,
        "num_coarse": 0,
        "num_fine": 8,
        "upsample_inv_s": 64.,
        "upsample_inv_s_factors": [16]
    })

    # Sphere tracing config: for waymo street scene
    if scene.main_class_name == "Street":
        model.ray_query_cfg = ConfigDict(
            query_mode="sphere_trace",
            query_param=ConfigDict(
                distance_scale=40.,
                min_step=.2,
                max_march_iters=500,
                drop_alive_rate=0.,
                tail_sample_threshold=20000,
                tail_sample_step_size=None,
                hit_threshold=.01
            ),
        )

    # Sphere tracing config: for single-object scene
    if scene.main_class_name == 'Main':
        model.ray_query_cfg = ConfigDict(
            query_mode="sphere_trace",
            query_param=ConfigDict(
                distance_scale=1.,
                min_step=.002,
                max_march_iters=500,
                drop_alive_rate=0.,
                tail_sample_threshold=10000,
                tail_sample_step_size=None,
                hit_threshold=1e-4
            ),
        )

    if args.regenerate_occgrid:
        resolution = (model.accel.occ.resolution * 2).tolist()
        ic(resolution)
        model.accel.occ = OccGridEma(
            resolution, occ_val_fn_cfg=ConfigDict(type='sdf', inv_s=256.0), occ_thre=0.8,
            init_cfg=ConfigDict(num_steps=128, num_pts=2**24, mode='from_net'))
        # model.accel.occ._init_from_net( num_steps=64, num_pts_per_batch=2**20)
        model.training_before_per_step(0)  # Run init_from_net

    model.training_before_per_step(args.training.num_iters)  # Set to last state
    # ray_query_cfg = ConfigDict(perturb=False, with_rgb=True, with_normal=False, nablas_has_grad=False)
    ray_query_cfg = ConfigDict(perturb=False, with_rgb=True,
                               with_normal=True, nablas_has_grad=False)
    ray_query_cfg.update(model.ray_query_cfg)

    cam0: Camera = scene.get_observer_groups_by_class_name('Camera', False)[0]

    # with torch.no_grad():
    #     cam0.intr.set_downscale(2)
    #     rays_o, rays_d = cam0.get_all_rays()
    #     (rays_o, *_), (rays_d, *_) = scene.convert_rays_in_nodes_list(rays_o, rays_d, [obj])

    # @torch.no_grad()
    # def to_img(tensor: torch.Tensor, H=cam0.intr.H, W=cam0.intr.W):
    #     *_, num_samples, channels = tensor.shape
    #     assert num_samples == H * W
    #     return tensor.view([H, W, channels]).data.cpu().numpy()

    # @torch.no_grad()
    # def render_fn(rays_o, rays_d, config):
    #     ray_input=dict(rays_o=rays_o, rays_d=rays_d, near=0.1,  far=120.0)
    #     ray_tested = model.space.ray_test(**ray_input)
    #     ret = model.ray_query(ray_input=ray_input, ray_tested=ray_tested, config=config, render_per_obj_individual=True)
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

    if args.downscale:
        cam0.intr.set_downscale(args.downscale)
    kaolin_camera = klCamera.from_args(view_matrix=w2c, focal_x=cam0.intr.focal()[0].item(),
                                       width=cam0.intr.W, height=cam0.intr.H, near=0.01, far=200.)
    state = ConfigDict(
        renderer=ConfigDict(
            antialiasing='none',
            clear_color_value=(0, 0, 0),
            canvas_height=cam0.intr.H,
            canvas_width=cam0.intr.W,
            target_fps=None,
            device=device,
            selected_camera=kaolin_camera,
            selected_camera_lens="perspective",
            reference_grids=['xz']
        ),
        nr_data_layers=ConfigDict()
    )
    if scene.image_embeddings is not None:
        image_embed_code = scene.image_embeddings[cam0.id](
            torch.scalar_tensor(args.slice_at, dtype=torch.long, device=device))
    else:
        image_embed_code = None
    neural_renderer = NR3DKaolinWispRenderer(model, ray_query_cfg, image_embed_code)
    app = APP("neuralsim v0.4.3", state, neural_renderer)
    print("Create window at resolution {}x{}".format(cam0.intr.W, cam0.intr.H))
    app.run()
