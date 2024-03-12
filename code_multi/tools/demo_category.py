"""
@file   demo_category.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Render demo for categorical assets.
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
import imageio
import numpy as np
from tqdm import tqdm
from datetime import datetime

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

import torch

from nr3d_lib.fmt import log
from nr3d_lib.checkpoint import sorted_ckpts
from nr3d_lib.maths import get_transform_np, normalize
from nr3d_lib.plot import color_depth, gallery
from nr3d_lib.config import ConfigDict, load_config
from nr3d_lib.utils import IDListedDict, cond_mkdir, import_str
from nr3d_lib.models.attributes import *

from app.models_env import PureColorSky
from app.resources import Scene, SceneNode, AssetBank, load_scene_bank
from app.resources.observers import Lidar, Camera
from app.renderers import BufferComposeRenderer

def main_function(args):
    device = torch.device('cuda')

    #---------------------------------------------
    #------------     Demo Scene     -------------
    #---------------------------------------------
    demo_scene = Scene('demo_scene', device=device)
    H, W, FOCAL = [float(i) for i in args.camera_hwf.split(',')]

    #---------------------------------------------
    #-----------       Demo Observer    ----------
    #---------------------------------------------
    camera = Camera('camera_0')
    camera.intr = PinholeCameraHWF(H=H, W=W, f=FOCAL)
    camera.near = 0.0
    camera.far = 8.0
    camera.to(device=device, dtype=torch.float32)

    # NOTE: From OpenCV Camera [+z forward, y down, x righr] to our assumed world (waymo/ros) Camera [+x forward, +z upward, y left]
    opencv_to_world = np.eye(4)
    opencv_to_world[:3, :3] = np.array(
        [[0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]])

    #*********************************************************************************************************************
    # NOTE: From our assumed world(waymo/ros) Camera [+x forward, +z upward, y left] to whatever dataset's defined camera convention.
    #       You can modify this if your dataset's convention is different from our assumed world (waymo/ros).
    #*********************************************************************************************************************
    world_to_dataset = np.eye(4)

    #---------------------------------------------
    #------     get objects with models     ------
    #---------------------------------------------
    demo_objs: IDListedDict[SceneNode] = IDListedDict()
    if args.from_neuralgen is None:
        #---------------------------------------------
        #------------- From neuralsim ----------------
        #---------------------------------------------
        exp_dir = args.exp_dir

        #---------------------------------------------
        #--------------     Load     -----------------
        if (ckpt_file:=args.get('load_pt', None)) is None:
            # Automatically load 'final_xxx.pt' or 'latest.pt'
            ckpt_file = sorted_ckpts(os.path.join(exp_dir, 'ckpts'))[-1]
        log.info("=> Use ckpt:" + str(ckpt_file))
        state_dict = torch.load(ckpt_file, map_location=device)

        #---------------------------------------------
        #-----------     Assetbank     ---------------
        # Only preserve the target class
        args.assetbank_cfg.class_name_cfgs = ConfigDict({args.class_name: args.assetbank_cfg.class_name_cfgs[args.class_name]})
        asset_bank = AssetBank(args.assetbank_cfg)
        scenebank_root = os.path.join(exp_dir, 'scenarios')
        scene_bank, scenebank_meta = load_scene_bank(scenebank_root, device=device)
        asset_bank.create_asset_bank(scene_bank, scenebank_meta, load_state_dict=state_dict['asset_bank'], device=device, class_name_list=f'Categorical.{args.class_name}')
        asset_bank.model_setup()
        for s in scene_bank:
            s.load_assets(asset_bank)
        # !!! Only call training_before_per_step when all assets are ready & loaded !
        asset_bank.training_before_per_step(args.training.num_iters)
        
        for s in scene_bank:
            for obj in s.get_drawable_groups_by_class_name(args.class_name, only_valid=False):
                obj.scale = obj.frame_data.subattr.scale[0]
                obj.i_valid = True
                obj.i_valid_flags = torch.tensor([True], dtype=torch.bool, device=device)
                obj.pose = TransformMat4x4() # put at center
                obj.to(device)
                demo_objs.append(obj)

        #---------------------------------------------
        #------------     Renderer     ---------------
        renderer = BufferComposeRenderer(args.renderer)
        renderer.populate(asset_bank)
        renderer.eval()
        asset_bank.eval()
        
        renderer.config.rayschunk = args.rayschunk
        renderer.config.with_normal = True
        renderer.config.with_env = False

        #---------------------------------------------
        #----     Camera initial pose (c2w)    -------
        #---- NOTE: in world (waymo/ros) convention, not in the dataset's world convention.
        #----       The conversion is done later.

        # NOTE: !!! from now on, everything is under world (waymo/ros) convention !!!
        # NOTE: in world (waymo/ros) convention:  we want looking at -x (from front to back) at the beginning.
        _rot0 = np.array(
                [[-1,0, 0], 
                [0, -1, 0],
                [0, 0, 1]]
        )
        # NOTE: in world (waymo/ros) convention: in front of the object (camera locates at +x position)
        _trans0 = np.array([[args.camera_distance], [0.], [0.]])
        c2w0_world = np.concatenate([np.concatenate([_rot0, _trans0], axis=-1), np.array([[0, 0, 0, 1]])], axis=0)
        c2w0_world = c2w0_world @ opencv_to_world
        # lidar = Lidar('lidar_0')

    else:
        # eg.
        #  python code_multi/tools/demo_category.py --from_neuralgen ~/ai_ws/neuralgen/logs/exp1 --camera_hwf 200,200,200
        #---------------------------------------------
        #------------- From neuralgen ----------------
        #---------------------------------------------
        exp_dir = os.path.join("./", "out", f"load_neuralgen_{os.path.split(args.from_neuralgen)[-1]}")

        #---------------------------------------------
        #--------------     Load     -----------------
        if (ckpt_file:=args.get('load_pt', None)) is None:
            # Automatically load 'final_xxx.pt' or 'latest.pt'
            ckpt_file = sorted_ckpts(os.path.join(args.from_neuralgen, 'ckpts'))[-1]
        log.info("=> Use ckpt:" + str(ckpt_file))
        state_dict = torch.load(ckpt_file, map_location=device)

        #---------------------------------------------
        #-------     Compatible configs     ----------
        neuralgen_config = load_config(os.path.join(args.from_neuralgen, 'config.yaml'))
        fg_cfg = neuralgen_config.fg
        # TODO: Temporarily applicable for StyleLoDNeuS; Consider aligning parts of the neuralgen and neuralsim code modules completely in the future.
        fg_cfg.target = "nr3d_lib.models.autodecoder.create_autodecoder"
        fg_cfg.param.framework.target = "nr3d_lib.models.fields_conditional.neus.style_lotd_neus.StyleLoTDNeuSModel"
        fg_ids = state_dict["scene"]["fg_net._keys"]
        scene_ids = [demo_scene.id] * len(fg_ids)
        fg_full_unique_ids = [f"{demo_scene.id}#{oid}" for oid in fg_ids]

        # truncation latent z
        if args.latent_tranc_ratio > 0: # e.g. 0.7
            z = state_dict["scene"]['fg_net._latents']['z.weight']
            avg_z = torch.mean(z, 0, keepdim=True) #.detach()
            state_dict["scene"]['fg_net._latents']['z.weight'] = z * (1-args.latent_tranc_ratio) + avg_z * args.latent_tranc_ratio

        fg_ray_query_cfg = neuralgen_config.renderer.param.common
        fg_ray_query_cfg.update(neuralgen_config.renderer.param.get('val', {}))
        fg_ray_query_cfg.update(fg_cfg.get('ray_query_cfg', {}))

        dummy_cfg = ConfigDict()
        dummy_cfg.class_name_cfgs = ConfigDict()
        # Load neuralgen configs as compatible with neuralsim configs
        dummy_cfg.class_name_cfgs[args.class_name] = fg_cfg
        dummy_cfg.class_name_cfgs['Sky'] = ConfigDict(target='app.models_env.PureColorSky', param={})

        #---------------------------------------------
        #-----------     Assetbank     ---------------
        asset_bank = AssetBank(dummy_cfg)
        asset_bank.drawable_shared_map = {args.class_name: list(zip(scene_ids, fg_ids))}
        fg_model_id = asset_bank.get_shared_model_id(args.class_name)
        fg_model = import_str(fg_cfg.target)(key_list=fg_full_unique_ids, cfg=fg_cfg.param)
        fg_model.ray_query_cfg = fg_ray_query_cfg
        asset_bank.add_module(fg_model_id, fg_model)

        fg_state_dict = dict()
        for k in ["_latents", "_models", "_keys", "_keys_prob"]:
            fg_state_dict[k] = state_dict["scene"][".".join(["fg_net", k])]
        dummy_state_dict = dict()
        dummy_state_dict[fg_model_id] = fg_state_dict

        if args.neuralgen_use_scale:
            fg_scale = state_dict['scene']['fg_scale.scale']

        # TODO: After achieving better alignment between neuralgen and neuralsim codes,
        #       consider loading assetbank into neuralgen as a class method in the future.
        asset_bank.load_asset_bank(dummy_state_dict)
        asset_bank.to(device)
        asset_bank.training_before_per_step(neuralgen_config.training.num_iters)

        for i, obj_id in enumerate(fg_ids):
            obj = SceneNode(obj_id, class_name=args.class_name)
            if args.neuralgen_use_scale:
                obj.scale = Scale(fg_scale[i])
            else:
                obj.scale = Scale()
            obj.i_valid = True
            obj.i_valid_flags = torch.tensor([True], dtype=torch.bool, device=device)
            obj.pose = Transform() # put at center
            obj.to(device)
            obj.model = fg_model
            demo_objs.append(obj)

        #---------------------------------------------
        #------------     Renderer     ---------------
        renderer_cfg = ConfigDict()
        renderer_cfg.common = ConfigDict(
            rayschunk=args.rayschunk,
            near=0.0,
            far=2.0*args.camera_distance,
            with_rgb=True,
            with_normal=True,
            with_env=True,
            perturb=False
        )
        renderer = BufferComposeRenderer(renderer_cfg)
        renderer.to(device)
        renderer.eval()
        asset_bank.eval()


        if args.neuralgen_use_convert_1:
            #---------------------------------------------
            #------  Coordinate convertion: from waymo/ros world to SRN world (+y forward, +x right, +z upward)
            world_to_dataset[:3,:3] = np.array(
                [[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]])
        elif args.neuralgen_use_convert_2:
            #---------------------------------------------
            #------  Coordinate convertion: from waymo/ros world to MVMC world (+z forward, +x left, +y upward)
            world_to_dataset[:3,:3] = np.array(
                [[0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]])
        else:
            pass

        #---------------------------------------------
        #----     Camera initial pose (c2w)    -------
        #---- NOTE: in world (waymo/ros) convention, not in the dataset's world convention.
        #----       The conversion is done later.

        # NOTE: !!! from now on, everything is under world (waymo/ros) convention !!!
        # NOTE: in world (waymo/ros) convention:  we want looking at -x (from front to back) at the beginning.
        _rot0 = np.array(
                [[-1,0, 0], 
                [0, -1, 0],
                [0, 0, 1]]
        )
        # NOTE: in world (waymo/ros) convention: in front of the object (camera locates at +x position)
        _trans0 = np.array([[args.camera_distance], [0.], [0.]])
        c2w0_world = np.concatenate([np.concatenate([_rot0, _trans0], axis=-1), np.array([[0, 0, 0, 1]])], axis=0)
        c2w0_world = c2w0_world @ opencv_to_world

    assert len(demo_objs) > 0, f"Found 0 instances of class_name={args.class_name}"
    
    #---------------------------------------------
    #----     White background / envmap     ------
    #---------------------------------------------
    base_objs = IDListedDict([])
    sky_obj = SceneNode(unique_id='env', class_name='Sky').to(device=device)
    sky_model_id = asset_bank.asset_compute_id(obj=sky_obj, scene=demo_scene)
    sky_model = PureColorSky(RGB=[255,255,255]).to(device=device)
    asset_bank.add_module(sky_model_id, sky_model)
    base_objs.append(sky_obj)
    
    demo_scene.add_node(camera)

    #---------------------------------------------
    #--------------     Plot     -----------------
    #---------------------------------------------
    expname = os.path.split(exp_dir.rstrip("/"))[-1]
    name = f"{args.mode}_{expname}_{args.outbase}.Categorical.{args.class_name}.{args.camera_path}"
    vid_root = os.path.join(args.exp_dir, args.dirname)
    cond_mkdir(vid_root)
    if args.save_raw:
        vid_raw_root = os.path.join(vid_root, name)
        cond_mkdir(vid_raw_root)
    def write_video(uri, frames, **kwargs):
        if len(frames) > 1:
            if ".mp4" not in uri:
                uri = f"{uri}.mp4"
            imageio.mimwrite(uri, frames, fps=args.fps, quality=args.quality, **kwargs)
            print(f"Video saved to {uri}")
        else:
            if ".mp4" in uri:
                uri = f"{os.path.splitext(uri)[0]}.png"
            imageio.imwrite(uri, frames[0], **kwargs)
            print(f"Image saved to {uri}")

    # NOTE: !!! from now on, every calculation is under world (waymo/ros) convention !!!
    c2w_trans0 = c2w0_world[:3,3]
    if args.camera_path == 'great_circle':
        # NOTE: world (waymo/ros) convention: "up" is +z
        up_vec = np.array([0., 0., 1.])
        # use the origin as the focus center
        focus_center = np.zeros([3])
        # key rotations of a spherical spiral path
        all_angles = np.linspace(0, 2*np.pi, args.num_views)
        # rotate about up vec
        rots = R.from_rotvec(all_angles[:, None] * up_vec[None, :])
        render_centers = rots.apply(c2w_trans0)
        # NOTE: world (waymo/ros) convention: "up" is +z, "forward" is +x
        render_c2ws = get_transform_np(render_centers, ox=(focus_center[None, :]-render_centers), oy=up_vec, preserve='x')
        
    elif args.camera_path == 'spherical_spiral':
        # the axis vertical to the small circle's area
        # NOTE: world (waymo/ros) convention: "up" is +z
        up_vec = np.array([0., 0., 1.])
        # use the origin as the focus center
        focus_center = np.zeros([3])
        # number of theta rounds.
        n_rots = 2.2
        # number of phi increasing.
        # up_angle = np.pi / 4.
        up_angle = np.deg2rad(args.camera_upangle_degree)
        
        # key rotations of a spherical spiral path
        sphere_thetas = np.linspace(0, np.pi * 2. * n_rots, args.num_views)
        sphere_phis = np.linspace(0, up_angle, args.num_views)
        
        # first rotate about up vec
        rots_theta = R.from_rotvec(sphere_thetas[:, None] * up_vec[None, :])
        render_centers = rots_theta.apply(c2w_trans0)
        # then rotate about horizontal vec
        horizontal_vec = normalize(np.cross(render_centers-focus_center[None, :], up_vec[None, :], axis=-1))
        rots_phi = R.from_rotvec(sphere_phis[:, None] * horizontal_vec)
        render_centers = rots_phi.apply(render_centers)
        
        # NOTE: world (waymo/ros) convention: "up" is +z, "forward" is +x
        render_c2ws = get_transform_np(render_centers, ox=(focus_center[None, :]-render_centers), oz=up_vec, preserve='x')
    elif args.camera_path == 'single_shot':
        render_c2ws = [c2w0_world @ opencv_to_world]
    else:
        raise NotImplementedError(f"camera_path={args.camera_path}")

    with torch.no_grad():
        if args.mode == 'instances':
            all_gather = dict()
            # for obj in demo_objs:
            #     all_gather[obj.id] = []

            log.info(f"Start [demo_category], mode={args.mode}, in {exp_dir}")
            for ind, obj in enumerate(tqdm(demo_objs, desc='rendering...')):
                
                # obj = demo_objs['segment-7670103006580549715_360_000_380_000_with_camera_labels#U122QiiD-P989MzfLjgsJg']
                demo_scene.load_from_nodes([obj] + base_objs.to_list())
                demo_scene.load_assets(asset_bank)
                
                frames = []
                
                for i in tqdm(range(args.num_views), disable=not args.progress, desc=f'obj[{ind}]'):
                    galleries = []
                    # NOTE: opencv_c2w = waymo_c2w @ opencv_cam_to_waymo_cam
                    #       ∵ opencv_c2w @ vecs_in_opencv = waymo_c2w @ opencv_cam_to_waymo_cam @ vecs_in_opencv
                    # NOTE: world_to_dataset converts our assumed world (waymo/ros) convention to the dataset's custom convention.
                    camera.world_transform = Transform(world_to_dataset @ render_c2ws[i] @ opencv_to_world).to(device=device, dtype=torch.float32)
                    
                    ret = renderer.render(demo_scene, observer=camera, render_per_obj_individual=False, show_progress=False)
                    rendered = ret['rendered']
                    def to_img(tensor):
                        return tensor.reshape([camera.intr.H, camera.intr.W, -1]).data.cpu().numpy()
                    rgb_volume = np.clip((to_img(rendered['rgb_volume'])*255).astype(np.uint8),0,255)
                    mask_volume = to_img(rendered['mask_volume'])
                    depth_volume = to_img(rendered['depth_volume'].unsqueeze(-1))
                    depth_volume = mask_volume * (depth_volume/renderer.config.far) + (1-mask_volume) * 1
                    depth_volume = color_depth(depth_volume.squeeze(-1), scale=1, cmap='turbo')    # turbo_r, viridis, rainbow

                    galleries.extend([rgb_volume, depth_volume])

                    # normals
                    if 'normals_volume' in rendered:
                        normals_volume = np.clip((to_img(rendered['normals_volume']/2+0.5)*255).astype(np.uint8),0,255)
                        galleries.append(normals_volume)
                    
                    galleries = gallery(np.array(galleries), ncols=1)
                    frames.append(galleries)
                    all_gather.setdefault(obj.id, []).append(galleries)
                if args.save_raw:
                    write_video(os.path.join(vid_raw_root, f"{obj.id}.mp4"), frames)
            
            total_frames = []
            for coll in zip(*all_gather.values()):
                total_fr = gallery(np.array(coll), nrows=max(int(np.sqrt(0.25*len(coll))),1))
                total_frames.append(total_fr)
            write_video(os.path.join(vid_root, f"{name}.mp4"), total_frames)

        elif args.mode == 'interpolation':
            all_gather = dict()
            for view_ind in range(args.num_views):
                all_gather[view_ind] = []

            latent_travel = [int(i) for i in args.latent_travel.split(',')]
            demo_obj_ind = latent_travel[0]
            demo_obj = demo_objs[demo_obj_ind]
            category_model_id = asset_bank.asset_compute_id(obj=demo_obj, scene=demo_scene)
            demo_scene.load_from_nodes([demo_obj] + base_objs.to_list())
            demo_scene.load_assets(asset_bank)
            
            latents = [asset_bank[category_model_id].get_latents(demo_objs[i].id)[args.latent_key].squeeze(0).data.cpu().numpy() for i in latent_travel]
            num_frames = (len(latents)-1) * args.num_frames_per_view
            interp = interp1d([i * args.num_frames_per_view for i in range(len(latents))], latents, axis=0)

            log.info(f"Start [demo_category], mode={args.mode}, in {exp_dir}")
            for view_ind, c2w in enumerate(tqdm(render_c2ws, desc='rendering...')):
                # obj = demo_objs['segment-7670103006580549715_360_000_380_000_with_camera_labels#U122QiiD-P989MzfLjgsJg']
                camera.world_transform = Transform(world_to_dataset @ c2w @ opencv_to_world).to(device=device, dtype=torch.float32)
                
                frames = []
                for i in tqdm(range(num_frames), disable=not args.progress, desc=f'view[{view_ind}]'):
                    latent = interp(i)
                    asset_bank[category_model_id]._latents[args.latent_key].weight[demo_obj_ind] = torch.from_numpy(latent).to(device=device, dtype=asset_bank.dtype)

                    galleries = []
                    # NOTE: opencv_c2w = waymo_c2w @ opencv_cam_to_waymo_cam
                    #       ∵ opencv_c2w @ vecs_in_opencv = waymo_c2w @ opencv_cam_to_waymo_cam @ vecs_in_opencv
                    # NOTE: world_to_dataset converts our assumed world (waymo/ros) convention to the dataset's custom convention.
                    
                    ret = renderer.render(demo_scene, observer=camera, render_per_obj_individual=False, show_progress=False)
                    rendered = ret['rendered']
                    def to_img(tensor):
                        return tensor.reshape([camera.intr.H, camera.intr.W, -1]).data.cpu().numpy()
                    rgb_volume = np.clip((to_img(rendered['rgb_volume'])*255).astype(np.uint8),0,255)
                    mask_volume = to_img(rendered['mask_volume'])
                    depth_volume = to_img(rendered['depth_volume'].unsqueeze(-1))
                    depth_volume = mask_volume * (depth_volume/renderer.config.far) + (1-mask_volume) * 1
                    depth_volume = color_depth(depth_volume.squeeze(-1), scale=1, cmap='turbo')    # turbo_r, viridis, rainbow

                    galleries.extend([rgb_volume, depth_volume])

                    # normals
                    if 'normals_volume' in rendered:
                        normals_volume = np.clip((to_img(rendered['normals_volume']/2+0.5)*255).astype(np.uint8),0,255)
                        galleries.append(normals_volume)
                    
                    galleries = gallery(np.array(galleries), ncols=1)
                    frames.append(galleries)
                    all_gather[view_ind].append(galleries)
                if args.save_raw:
                    write_video(os.path.join(vid_raw_root, f"{view_ind}.mp4"), frames)
            
            total_frames = []
            for coll in zip(*all_gather.values()):
                total_fr = gallery(np.array(coll), nrows=int(np.sqrt(0.25*len(coll))))
                total_frames.append(total_fr)
            write_video(os.path.join(vid_root, f"{name}.mp4"), total_frames)

        else:
            raise ValueError(f"Invalid mode=={args.mode}")

if __name__ == "__main__":
    from nr3d_lib.config import BaseConfig
    bc = BaseConfig()
    bc.parser.add_argument("--mode", type=str, default="instances")

    bc.parser.add_argument("--class_name", type=str, default="Vehicle")
    bc.parser.add_argument("--progress", action='store_true', help="If set, shows per frame progress.")
    bc.parser.add_argument("--save_raw", action='store_true')
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--fps", type=int, default=15)
    bc.parser.add_argument("--quality", type=int, default=None, help="Sets the quality for imageio.mimwrite (range: 0-10; 10 is the highest; default is 5).")
    bc.parser.add_argument("--dirname", type=str, default='videos', help="Sets the output directory to /path/to/exp_dir/${dirname}`.")
    bc.parser.add_argument("--outbase", type=str, default=datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S"), help="Sets the basename of the output file (without extension).")
    bc.parser.add_argument("--rayschunk", type=int, default=10000)
    
    bc.parser.add_argument("--camera_hwf", type=str, default="100,100,100", help="camera image h, w and focal")
    bc.parser.add_argument("--camera_path", type=str, default="spherical_spiral", help="choose between [spiral, small_circle, great_circle, spherical_spiral]")
    bc.parser.add_argument("--camera_distance", type=float, default=1.3) # For waymo, set to 6.0; for SRN, set to 1.3; for MVMC, set to 3.0
    bc.parser.add_argument("--camera_upangle_degree", type=float, default=45.0) # For waymo, set to 6.0; for SRN, set to 1.3; for MVMC, set to 3.0
    bc.parser.add_argument("--num_views", type=int, default=60, help="")

    bc.parser.add_argument("--from_neuralgen", type=str, default=None)
    bc.parser.add_argument("--neuralgen_use_scale", action='store_true')
    bc.parser.add_argument("--neuralgen_use_convert_1", action='store_true')
    bc.parser.add_argument("--neuralgen_use_convert_2", action='store_true')

    # For mode == 'interpolation'
    bc.parser.add_argument("--latent_travel", type=str, default='0,1,2,3,4,5,6,7,8,9', help="interpolation end")
    bc.parser.add_argument("--latent_key", type=str, default='z', help="interpolation latent key")
    bc.parser.add_argument("--latent_tranc_ratio", type=float, default=0, help="e.g. 0.7 to use. default disabled with 0. value")
    bc.parser.add_argument("--num_frames_per_view", type=int, default=45, help="")
    # NOTE: better set args.num_views to be smaller in this mode. e.g. 16

    main_function(bc.parse(stand_alone=False))