"""
@file   custom_autodrive_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset IO for custom autonoumous driving datasets
"""
import os
import pickle
import numpy as np
from glob import glob
from typing import Any, Dict, List, Literal
from scipy.spatial.transform import Rotation as R

from nr3d_lib.utils import load_rgb
from nr3d_lib.config import ConfigDict

from dataio.scene_dataset import SceneDataset
from dataio.utils import clip_node_data, clip_node_segments
from dataio.autonomous_driving.custom.filter_dynamic import stat_dynamic_objects

def idx_to_frame_str(frame_index):
    return f'{frame_index:08d}'

def idx_to_img_filename(frame_index):
    return f'{idx_to_frame_str(frame_index)}.jpg'

def idx_to_lidar_filename(frame_index):
    return f'{idx_to_frame_str(frame_index)}.npz'

def idx_to_mask_filename(frame_index, compress=True):
    ext = 'npz' if compress else 'npy'
    return f'{idx_to_frame_str(frame_index)}.{ext}'

class CustomAutoDriveDataset(SceneDataset):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.populate(**config)
    
    def populate(
        self, 
        root: str, 
        image_dirname: str = 'images', 
        lidar_dirname: str = 'lidars', 
        mask_dirname: str = 'masks', 
        mask_taxonomy: Literal['cityscapes', 'ade20k'] = 'cityscapes', 
        mono_depth_dirname: str = "depths", 
        mono_normals_dirname: str = "normals", 
        ):
        self.main_class_name = "Street"
        self.root = root
        self.image_dirname = image_dirname
        self.mono_depth_dirname = mono_depth_dirname
        self.mono_normals_dirname = mono_normals_dirname
        self.lidar_dirname = lidar_dirname

        """
        An example dataset's world convention:
        
        facing [+x] direction, z upwards, y left
                z ↑ 
                  |  ↗ x
                  | /
                  |/
         ←--------o
        y
        """
        # Converts vectors in OpenCV' coords to datasets'
        opencv_to_world = np.eye(4)
        opencv_to_world[:3 ,:3] = np.array([[0, 0, 1],[-1, 0, 0],[0, -1, 0]])
        self.opencv_to_world = opencv_to_world

        self._populate_mask_settings(mask_dirname=mask_dirname, mask_taxonomy=mask_taxonomy)

    def _populate_mask_settings(
        self, 
        mask_dirname: str = "masks", 
        mask_taxonomy: Literal['cityscapes', 'ade20k'] = 'cityscapes', ):
        self.mask_dirname = mask_dirname
        self.mask_taxonomy = mask_taxonomy
        # Taxonomy reference source: mmseg/core/evaluation/class_names.py
        if self.mask_taxonomy == 'cityscapes':
            #---------------- Cityscapes semantic segmentation
            self.semantic_classes = [
                'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                'bicycle'
            ]
            
            self.semantic_dynamic_classes = [
                'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
            ]
            self.semantic_free_space_classes = ['sky']
            self.semantic_human_classes = ['person', 'rider']
            self.semantic_road_classes = ['road']
            
            self.dataset_classes_in_sematic = {
                'unknwon': ['train'],
                'Vehicle': ['car', 'truck', 'bus'],
                'Pedestrian': ['person'],
                'Sign': ['traffic light', 'traffic sign'],
                'Cyclist': ['rider', 'motorcycle', 'bicycle']
            }
        
        elif self.mask_taxonomy == 'ade20k':
            #---------------- ADE20k semantic segmentation
            self.semantic_classes = [
                'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
                'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
                'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
                'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
                'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
                'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
                'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
                'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
                'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
                'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
                'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
                'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
                'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
                'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
                'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
                'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
                'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
                'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
                'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
                'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                'clock', 'flag'
            ]
                        
            self.semantic_dynamic_classes = [
                'person', 'car', 'bus', 'truck', 'van', 'boat', 'airplane', 'ship', 
                'minibike', 'animal', 'bicycle'
            ]
            self.semantic_free_space_classes = ['sky']
            self.semantic_human_classes = ['person']
            self.semantic_road_classes = ['road']
            
            self.dataset_classes_in_sematic = {
                'unknwon': ['train'],
                'Vehicle': ['car', 'bus', 'truck', 'van'],
                'Pedestrian': ['person'],
                'Sign': ['traffic light'],
                'Cyclist': ['minibike', 'bicycle']
            }
        
        else:
            raise RuntimeError(f"Invalid mask_taxonomy={mask_taxonomy}")

        self.semantic_classes_ind_map = {cn: i for i, cn in enumerate(self.semantic_classes)}

    def _get_scenario(
        self, scenario: dict, # The original scenario loaded from preprocessed dataset
        *, 
        observer_cfgs: dict, # scene_bank's per class_name observer configs
        object_cfgs: dict = {}, # scene_bank's per class_name object configs
        no_objects=False, # Set to [true] to load no object at all.
        align_orientation=False, # Set to [true] to rotate the street obj to align with major vehicle moving direction
        camera_front_name='camera_FRONT', 
        camera_model='pinhole', # [pinhole, opencv, fisheye]; respected in Scene.load_from_scenario.load_observers
        aabb_extend: float = 60.0, # Not actually used for now.
        start=None, stop=None, # (Optionally) Drop beggining frames or ending frames that we do not want.
        ):

        #------------------------------------------------------
        #--------    Dynamic object statistics     ------------
        #------------------------------------------------------
        if 'dynamic_stats' not in scenario['metas'].keys():
            # NOTE: 0.3 is relatively over head
            scenario['metas']['dynamic_stats'] = stat_dynamic_objects(scenario, loc_eps=0.3)

        new_scenario = dict(scene_id=scenario['scene_id'], metas=scenario['metas'])
        data_frame_offset = new_scenario['metas']['data_frame_offset'] = start if start is not None else 0
        original_num_frames = scenario['metas']['n_frames']
        num_frames = new_scenario['metas']['n_frames'] = (stop if stop is not None else original_num_frames) - data_frame_offset
        # new_scenario['metas']['ego_pose'] = new_scenario['metas']['ego_pose'][start_frame:start_frame+n_frames]
        
        new_scene_objects = new_scenario['objects'] = dict()
        #------------------------------------------------------
        #------------------     Street      -------------------
        #------------------------------------------------------
        street_odict = dict(id=f"street", class_name=self.main_class_name)
        new_scene_objects[street_odict['id']] = street_odict
        # The major class_name that we focus on in street-view surface reconstruction task
        new_scenario['metas']['main_class_name'] = self.main_class_name
        
        #------------------------------------------------------
        #------------------     Objects      ------------------
        #------------------------------------------------------
        obj_box_list_per_frame = dict()
        obj_box_list_per_frame_dynamic_only = dict()
        
        category_mapping = {
            'Car': 'Vehicle', 
            'Pedestrian': 'Pedestrian', 
            'Bicycle': 'Cyclist', 
            'Van': 'Vehicle', 
            'Cyclist': 'Cyclist', 
            'Bus': 'Vehicle'
        }
        
        # Merge dynamic stats
        new_dyna_stats = {}
        for cls_name, cls_dict in scenario['metas']['dynamic_stats'].items():
            if cls_name not in category_mapping.keys():
                continue
            new_cls_name = category_mapping[cls_name]
            new_dyna_stats.setdefault(new_cls_name, {'is_dynamic': [], 'n_dynamic': 0, 'by_loc': [], 'by_speed': []})
            new_dyna_stats[new_cls_name]['is_dynamic'].extend(cls_dict.get('is_dynamic', []))
            new_dyna_stats[new_cls_name]['by_loc'].extend(cls_dict.get('by_loc', []))
            new_dyna_stats[new_cls_name]['by_speed'].extend(cls_dict.get('by_speed', []))
            new_dyna_stats[new_cls_name]['n_dynamic'] += cls_dict.get('n_dynamic', 0)
        scenario['metas']['dynamic_stats'] = new_dyna_stats
        
        for oid, odict in scenario['objects'].items():
            # !!! Fix
            if odict['class_name'] not in category_mapping.keys():
                continue
            o_class_name = odict['class_name'] = category_mapping[odict['class_name']]
            
            #---- Scene meta data for objects (bbox list of each frame)
            obj_box_list_per_frame.setdefault(o_class_name, [[] for _ in range(original_num_frames)])
            obj_box_list_per_frame_dynamic_only.setdefault(o_class_name, [[] for _ in range(original_num_frames)])
            for seg in odict['segments']:
                for seg_local_fi in range(seg['n_frames']):
                    fi = seg['start_frame'] + seg_local_fi
                    
                    # !!! Fix
                    seg['data']['global_frame_inds'] = np.arange(seg['start_frame'], seg['start_frame']+seg['n_frames'], 1)
                    
                    # transform_in_world (12) + scale (3)
                    cur_box = np.concatenate([seg['data']['transform'][seg_local_fi][:3, :].reshape(-1), seg['data']['scale'][seg_local_fi]])
                    obj_box_list_per_frame[o_class_name][fi].append(cur_box)
                    if oid in scenario['metas']['dynamic_stats'][o_class_name]['is_dynamic']:
                        obj_box_list_per_frame_dynamic_only[o_class_name][fi].append(cur_box)
            
            if no_objects:
                continue
            
            #---- Load objects to scenario
            if o_class_name not in object_cfgs.keys():
                # Ignore un-wanted class_names
                continue
            cfg = object_cfgs[o_class_name]
            if cfg.get('dynamic_only', False) and (oid not in scenario['metas']['dynamic_stats'][o_class_name]['is_dynamic']):
                # Ignore non dynamic objects when set dynamic_only
                continue
            #---- Clip according to start, stop
            odict = clip_node_segments(odict, start, stop)
            if len(odict['segments']) == 0:
                # Ignore empty nodes after clip
                continue
            new_scene_objects[oid] = odict
        
        for class_name, box_list_per_frame in obj_box_list_per_frame.items():
            box_list_per_frame = [(np.stack(b, axis=0) if len(b) > 0 else []) for b in box_list_per_frame]
            obj_box_list_per_frame[class_name] = box_list_per_frame
        new_scenario['metas']['obj_box_list_per_frame'] = obj_box_list_per_frame
        
        for class_name, box_list_per_frame in obj_box_list_per_frame_dynamic_only.items():
            box_list_per_frame = [(np.stack(b, axis=0) if len(b) > 0 else []) for b in box_list_per_frame]
            obj_box_list_per_frame_dynamic_only[class_name] = box_list_per_frame
        new_scenario['metas']['obj_box_list_per_frame_dynamic_only'] = obj_box_list_per_frame_dynamic_only
        
        #------------------------------------------------------
        #-----------------     Observers      -----------------
        #------------------------------------------------------
        new_scene_observers = new_scenario['observers'] = dict()
        
        #------------------------------------------------------
        #------------------     Ego Car      ------------------
        #------------------------------------------------------
        # frame_pose = scenario['observers']['ego_car']['data']['v2w']
        
        cam_intrs_all = []
        cam_c2ws_all = []
        cam_front_c2ws_all = []
        
        #------------------------------------------------------
        #------------------     Cameras      ------------------
        #------------------------------------------------------
        if 'Camera' in observer_cfgs.keys():
            new_scenario['metas']['cam_id_list'] = observer_cfgs['Camera'].list
        for oid, odict in scenario['observers'].items():
            if (o_class_name:=odict['class_name']) == 'Camera':
                hw = odict['data']['hw']
                #---- Opt1: [4,] fx, fy, cx, cy
                # intr = np.tile(np.eye(3), [original_num_frames,1,1])
                # intr[:,0,0], intr[:,1,1], intr[:,0,2], intr[:,1,2] = odict['data']['intr'][:,:4].T
                #---- Opt2: [3,3]
                intr = odict['data']['intr'][..., :3, :3]
                
                c2w = odict['data']['c2w']
                # global_timestamps = odict['data']['global_timestamps']
                # global_frame_inds = odict['data']['global_frame_inds']
                global_frame_inds = np.arange(original_num_frames)
                
                cam_intrs_all.append(intr)
                cam_c2ws_all.append(c2w)
                
                if oid == camera_front_name:
                    cam_front_c2ws_all.append(c2w)
                
                if o_class_name not in observer_cfgs.keys():
                    # Ignore un-wanted class_names
                    continue
                if oid not in observer_cfgs[o_class_name].list:
                    # Ignore un-wanted oberver id
                    continue
                
                new_odict = dict(
                    class_name='Camera', n_frames=odict['n_frames'], 
                    camera_model=camera_model, 
                    data=dict(
                        # global_timestamps=global_timestamps, 
                        global_frame_inds=global_frame_inds, 
                        hw=hw, intr=intr, transform=c2w
                    )
                )
                if camera_model != 'pinhole':
                    assert RuntimeError(f"camera_model={camera_model} expects `distortion` parameter")
                    distortion = odict['data']['distortion']
                    new_odict['data']['distortion'] = distortion
                    
                #---- Clip according to start, stop
                new_scene_observers[oid] = clip_node_data(new_odict, start, stop)

        #------------------------------------------------------
        #-------------------     Lidars      ------------------
        #------------------------------------------------------
        if 'RaysLidar' in observer_cfgs.keys():
            new_scenario['metas']['lidar_id_list'] = observer_cfgs['RaysLidar'].list
        for oid, odict in scenario['observers'].items():
            if (o_class_name:=odict['class_name']) == 'RaysLidar':
                # l2v = odict['data']['l2v']
                # global_timestamps = odict['data']['global_timestamps']
                # global_frame_inds = odict['data']['global_frame_inds']
                global_frame_inds = np.arange(original_num_frames)
                
                if o_class_name not in observer_cfgs.keys():
                    # Ignore un-wanted class_names
                    continue
                if oid not in observer_cfgs[o_class_name].list:
                    # Ignore un-wanted oberver id
                    continue

                new_odict = dict(
                    class_name='RaysLidar', n_frames=odict['n_frames'], 
                    data=dict(
                        # global_timestamps=global_timestamps, 
                        global_frame_inds=global_frame_inds, 
                        transform=np.tile(np.eye(4), [original_num_frames,1,1])
                        # transform=frame_pose @ l2v
                    )
                )
                new_scene_observers[oid] = clip_node_data(new_odict, start, stop)

        #------------------------------------------------------
        #---------------     Other meta infos      ------------
        #------------------------------------------------------
        if start is None: start = 0
        if stop is None or stop == -1: stop = scenario['metas']['n_frames']
        
        # cam_intrs_all = np.array(cam_intrs_all).reshape(-1, 3, 3)
        # cam_c2ws_all = np.array(cam_c2ws_all).reshape(-1, 4, 4)
        cam_front_c2ws_all = np.array(cam_front_c2ws_all).reshape(-1, 4, 4)[start:stop]
        
        # NOTE: Convert original [camera<openCV> to world<self>] to [camera<self> to world<self>] 
        #       (i.e. to get camera's poses in world)
        world_to_opencv = np.linalg.inv(self.opencv_to_world)
        cam_front_poses_in_world = cam_front_c2ws_all @ world_to_opencv[None, ...]
        cam_front_tracks_in_world = cam_front_poses_in_world[:, :3, 3]
        
        # NOTE: Average rotation about z axis.
        r = R.from_matrix(cam_front_poses_in_world[:, :3, :3])
        avg_rot = r.mean()
        avg_rot_zyx = avg_rot.as_euler('zyx', degrees=False)
        avg_rot_z: float = avg_rot_zyx[0]
        avg_rot_z_mat = R.from_rotvec(np.array([0, 0, avg_rot_z])).as_matrix()
        new_scenario['metas']['average_rot_z'] = avg_rot_z
        new_scenario['metas']['average_rot_mat'] = avg_rot_z_mat
        
        # NOTE: Test whether the average rotation matrix works
        # bg_obj_o2w = np.eye(4)
        # bg_obj_o2w[:3, :3] = avg_rot_z_mat
        # bg_obj_w2o = np.linalg.inv(bg_obj_o2w)
        # # Should be points to x by average, with z untouched.
        # cam_front_tracks_in_bg_obj = np.einsum('ij,nj->ni', bg_obj_w2o, cam_front_c2ws_all[:, :, 3])[:, :3]
        
        if align_orientation:
            street_transform = np.tile(np.eye(4)[None,...], (new_scenario['metas']['n_frames'],1,1))
            street_transform[:, :3, :3] = new_scenario['metas']['average_rot_mat']
            
            street_odict["n_frames"] = new_scenario['metas']["n_frames"]
            street_odict["data"] = dict(transform=street_transform)
        
        # NOTE: Just a rough calculation, might not be used.
        aabb_min = cam_front_tracks_in_world.min(axis=0) - aabb_extend
        aabb_max = cam_front_tracks_in_world.max(axis=0) + aabb_extend
        new_scenario['metas']['aabb'] = np.stack([aabb_min, aabb_max], axis=0)

        return new_scenario

    def get_scenario_fpath(self, scene_id: str) -> str:
        fpath = os.path.join(self.root, scene_id, f"scenario.pt")
        assert os.path.exists(fpath), f'Not exist: {fpath}'
        return fpath

    def get_scenario(self, scene_id: str, **kwargs) -> Dict[str, Any]:
        scenario_fpath = self.get_scenario_fpath(scene_id)
        with open(scenario_fpath, 'rb') as f:
            scenario = pickle.load(f)
        # return scenario
        return self._get_scenario(scenario, **kwargs)

    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = os.path.join(self.root, scene_id, self.image_dirname, camera_id, idx_to_img_filename(frame_index))
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        return load_rgb(fpath)

    def get_image_mono_depth(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = os.path.join(self.root, scene_id, self.image_mono_depth_dirname, camera_id, f'{idx_to_frame_str(frame_index)}.npz')
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        depth = np.load(fpath)['arr_0'].astype(np.float32)
        return depth

    def get_image_mono_normals(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = os.path.join(self.root, scene_id, self.image_mono_normals_dirname, camera_id, f'{idx_to_frame_str(frame_index)}.jpg')
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        # [-1.,1.] np.float32 in OpenCV local coords
        normals = load_rgb(fpath)*2-1
        return normals

    def get_raw_mask(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True):
        fpath = os.path.join(self.root, scene_id, self.mask_dirname, camera_id, idx_to_mask_filename(frame_index, compress=compress))
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        # [H, W]
        if compress:
            arr = np.load(fpath)['arr_0']
        else:
            arr = np.load(fpath)
        return arr
    def get_image_occupancy_mask(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True) -> np.ndarray:
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.ones_like(raw).astype(np.bool8)
        for cls in self.semantic_free_space_classes:
            ret[raw==self.semantic_classes_ind_map[cls]] = False
        # [H, W] 
        # Binary occupancy mask on RGB image. 1 for occpied, 0 for not.
        return ret.squeeze()
    def get_image_semantic_mask_by_type(
        self, scene_id: str, camera_id: str, 
        sem_type: Literal['dynamic', 'human', 'road', 'anno_dontcare'], 
        frame_index: int, *, compress=True) -> np.ndarray:
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.zeros_like(raw).astype(np.bool8)
        if sem_type == 'dynamic':
            for cls in self.semantic_dynamic_classes:
                ind = self.semantic_classes_ind_map[cls]
                ret[raw==ind] = True
        elif sem_type == 'human':
            for cls in self.semantic_human_classes:
                ind = self.semantic_classes_ind_map[cls]
                ret[raw==ind] = True
        elif sem_type == 'road':
            for cls in self.semantic_road_classes:
                ind = self.semantic_classes_ind_map[cls]
                ret[raw==ind] = True
        else:
            raise RuntimeError(f"Invalid sem_type={sem_type}")
        # Binary semantic mask on RGB image. 1 for matched, 0 for not.
        return ret.squeeze()

    def get_lidar(self, scene_id: str, lidar_id: str, frame_index: int) -> Dict[str, np.ndarray]:
        fpath = os.path.join(self.root, scene_id, self.lidar_dirname, lidar_id, idx_to_lidar_filename(frame_index))
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        arr_dict = np.load(fpath)
        # TODO: There is room for a more clever design here: 
        #       When it's not a TOP lidar, we could just store azimuth and inclinations, and then calculate them just before this get function returns.
        return dict(rays_o=arr_dict['rays_o'], rays_d=arr_dict['rays_d'], ranges=arr_dict['ranges'])

if __name__ == "__main__":
    def test_distort():
        import torch
        import imageio
        from nr3d_lib.utils import check_to_torch, cond_mkdir
        from kornia.geometry.calibration.undistort import undistort_image
        device=torch.device('cuda')
        dataset_cfg_param = ConfigDict(
            root='/data1/multimodel_data_ailab'
        )
        scenebank_cfg = ConfigDict(
            # scenarios=['2022.08.19_17.58.37_scene'], 
            observer_cfgs=ConfigDict(
                Camera=ConfigDict(
                    list=['camera_sv_front', 'camera_sv_left', 
                          'camera_sv_right', 'camera_sv_rear']
                ), 
                RaysLidar=ConfigDict(
                    list=['lidar_middle']
                ), 
            ), 
            object_cfgs=ConfigDict(
                Vehicle=ConfigDict(
                    dynamic_only=False
                )
            ), 
            no_objects=False, 
            align_orientation=False, 
            distortion_model='fisheye'
        )
        dataset = CustomAutoDriveDataset(dataset_cfg_param)
        
        scene_id = '2022.08.19_17.58.37_scene'
        cam_id = 'camera_sv_left'
        # frame_ind = 83
        frame_ind = 159
        scenario = dataset.get_scenario(scene_id, **scenebank_cfg)
        odict = scenario['observers'][cam_id]
        
        #---------------- OpenCV camera model
        # K = check_to_torch(odict['data']['intr'][frame_ind], device=device, dtype=torch.float)
        # dist = check_to_torch(odict['data']['distortion'][frame_ind], device=device, dtype=torch.float)
        # rgb = dataset.get_image(scene_id, cam_id, frame_ind)
        # # [H, W, C] -> [C, H, W]
        # rgb_tensor = check_to_torch(rgb, device=device, dtype=torch.float).movedim(-1, 0)
        # rgb0_tensor = undistort_image(rgb_tensor, K, dist)
        # # {C, H, W} -> [H, W, C]
        # rgb0 = rgb0_tensor.movedim(0, -1).contiguous().data.cpu().numpy()
        # cond_mkdir('./dev_test/test_distortion_custom')
        # imageio.imwrite(f'./dev_test/test_distortion_custom/frame_{frame_ind}.png', rgb)
        # imageio.imwrite(f'./dev_test/test_distortion_custom/frame_{frame_ind}_undist.png', rgb0)
        
        #---------------- Fisheye camera model
        import cv2
        K = odict['data']['intr'][frame_ind]
        dist = odict['data']['distortion'][frame_ind][:4]
        # rgb0 = cv2.fisheye.undistortImage(rgb, K, dist)
        rgb = dataset.get_image(scene_id, cam_id, frame_ind)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, dist, np.eye(3), K, rgb.shape[:2][::-1], cv2.CV_16SC2) # CV_16SC2: 2 channel 16bit integer
        rgb0 = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        imageio.imwrite(f'./dev_test/test_distortion_custom/frame_{frame_ind}.png', rgb)
        imageio.imwrite(f'./dev_test/test_distortion_custom/frame_{frame_ind}_undist.png', rgb0)
    
    def test_scenario():
        import torch
        device=torch.device('cuda')
        from icecream import ic
        from nr3d_lib.utils import import_str
        from nr3d_lib.config import ConfigDict
        from app.resources import create_scene_bank
        from dataio.data_loader import SceneDataLoader
        
        dataset_cfg = ConfigDict(
            target='dataio.autonomous_driving.CustomAutoDriveDataset', 
            param=ConfigDict(
                root='/data1/multimodel_data_ailab'
            )
        )
        scenebank_cfg = ConfigDict(
            scenarios=['2022.08.19_17.58.37_scene'], 
            observer_cfgs=ConfigDict(
                Camera=ConfigDict(
                    list=['camera_sv_front', 'camera_sv_left', 
                          'camera_sv_right', 'camera_sv_rear']
                ), 
                RaysLidar=ConfigDict(
                    list=['lidar_middle']
                ), 
            ), 
            object_cfgs=ConfigDict(
                Vehicle=ConfigDict(
                    dynamic_only=False
                )
            ), 
            no_objects=False, 
            align_orientation=False, 
            camera_front_name='camera_sv_front', 
            camera_model='fisheye'
        )
        dataset_impl: SceneDataset = import_str(dataset_cfg.target)(dataset_cfg.param)
        scene_bank, _ = create_scene_bank(
            dataset=dataset_impl, device=device, 
            scenebank_root=None,
            scenebank_cfg=scenebank_cfg, 
            drawable_class_names=['Street', 'Distant', 'Vehicle', 'Pedestrian'], 
            misc_node_class_names=['node', 'EgoVehicle'], 
        )
        scene = scene_bank[0]
        # scene.debug_vis_multi_frame(40)
        scene_dataloader = SceneDataLoader(
            scene_bank, dataset_impl, device=device, 
            config=ConfigDict(
                preload=False, 
                tags=ConfigDict(
                    camera=ConfigDict(
                        downscale=1, 
                        list=scenebank_cfg.observer_cfgs.Camera.list
                    ), 
                    lidar=ConfigDict(
                        list=scenebank_cfg.observer_cfgs.RaysLidar.list, 
                        multi_lidar_merge=True, 
                        filter_kwargs=ConfigDict(
                            filter_valid=True, 
                            filter_in_cams=False, 
                            filter_out_objs=False
                        )
                    )
                )
            ))
        scene.debug_vis_anim(
            scene_dataloader=scene_dataloader, 
            plot_image=True, camera_length=0.6, 
            plot_lidar=True, lidar_pts_downsample=4)
    test_scenario()
    # test_distort()