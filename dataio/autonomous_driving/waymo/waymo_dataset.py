"""
@file   waymo_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset IO for Waymo Open Dataset - Perception v1
"""
import os
import sys
import pickle
import numpy as np
from glob import glob
from typing import Any, Dict, List
from scipy.spatial.transform import Rotation as R

from nr3d_lib.utils import load_rgb
from nr3d_lib.config import ConfigDict

from dataio.dataset_io import DatasetIO
from dataio.utils import clip_node_data, clip_node_segments

#---------------- Waymo original definition
WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
WAYMO_CAMERAS = ['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT']
WAYMO_LIDARS = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']
# NUM_CAMERAS = len(WAYMO_CAMERAS)
# NUM_LIDARS = len(WAYMO_LIDARS)

#---------------- Cityscapes semantic segmentation
cityscapes_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]
cityscapes_classes_ind_map = {cn: i for i, cn in enumerate(cityscapes_classes)}

cityscapes_dynamic_classes = [
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

cityscapes_human_classes = [
    'person', 'rider'
]

waymo_classes_in_cityscapes = {
    'unknwon': ['train'],
    'Vehicle': ['car', 'truck', 'bus'],
    'Pedestrian': ['person'],
    'Sign': ['traffic light', 'traffic sign'],
    'Cyclist': ['rider', 'motorcycle', 'bicycle']
}

#-----------------------------------------------------
#------- Common functions used by creation and loading
#-----------------------------------------------------
def idx_to_camera_id(camera_index):
    # return f'camera_{camera_index}'
    return f'camera_{WAYMO_CAMERAS[camera_index]}'

def idx_to_frame_str(frame_index):
    return f'{frame_index:08d}'

def idx_to_img_filename(frame_index):
    return f'{idx_to_frame_str(frame_index)}.jpg'

def idx_to_mask_filename(frame_index, compress=True):
    ext = 'npz' if compress else 'npy'
    return f'{idx_to_frame_str(frame_index)}.{ext}'

def idx_to_lidar_id(lidar_index):
    # return f'lidar_{lidar_index}'
    return f'lidar_{WAYMO_LIDARS[lidar_index]}'

def idx_to_lidar_filename(frame_index):
    return f'{idx_to_frame_str(frame_index)}.npz'

def file_to_scene_id(fname):
    return os.path.splitext(os.path.basename(os.path.normpath(fname)))[0]

def parse_seq_file_list(root: str, seq_list_fpath: str = None, seq_list: List[str] = None):
    assert os.path.exists(root), f'Not exist: {root}'
    
    if seq_list is None and seq_list_fpath is not None:
        with open(seq_list_fpath, 'r') as f:
            seq_list = f.read().splitlines()
    
    if seq_list is not None:
        seq_list = [s.split(',')[0].rstrip(".tfrecord") for s in seq_list]
        seq_fpath_list = []
        for s in seq_list:
            seq_fpath = os.path.join(root, f"{s}.tfrecord")
            assert os.path.exists(seq_fpath), f"Not exist: {seq_fpath}"
            seq_fpath_list.append(seq_fpath)
    else:
        seq_fpath_list = list(sorted(glob(os.path.join(root, "*.tfrecord"))))
    
    assert len(seq_fpath_list) > 0, f'No matching .tfrecord found in: {root}'
    return seq_fpath_list

#-----------------------------------------------------
#------- Dataset IMPL (implement standard APIs)
#-----------------------------------------------------
class WaymoDataset(DatasetIO):
    def __init__(self, config: ConfigDict) -> None:
        self.config = config
        self.populate(**config)

    def populate(
        self, root: str, 
        rgb_dirname: str = "images", 
        lidar_dirname: str = "lidars", 
        mask_dirname: str = "masks", 
        rgb_mono_depth_dirname: str = "depths", 
        rgb_mono_normals_dirname: str = "normals", 
        pcl_dirname: str = None, 
        ):

        assert os.path.exists(root), f"Not exist: {root}"

        self.main_class_name = "Street"
        self.root = root
        
        self.rgb_dirname = rgb_dirname
        self.rgb_mono_depth_dirname = rgb_mono_depth_dirname
        self.rgb_mono_normals_dirname = rgb_mono_normals_dirname
        self.lidar_dirname = lidar_dirname
        self.pcl_dirname = pcl_dirname
        self.mask_dirname = mask_dirname

    @property
    def up_vec(self) -> np.ndarray:
        return np.array([0., 0., 1.])   

    @property
    def forward_vec(self) -> np.ndarray:
        return np.array([1., 0., 0.])
    
    @property
    def right_vec(self) -> np.ndarray:
        return np.array([0., -1., 0.])

    def get_all_available_scenarios(self) -> List[str]:
        scenario_file_list = list(sorted(glob(os.path.join(self.root, "*", "scenario.pt"))))
        scenario_list = [os.path.splitext(os.path.basename(s))[0] for s in scenario_file_list]
        return scenario_list

    def get_scenario_fpath(self, scene_id: str) -> str:
        fpath = os.path.join(self.root, scene_id, "scenario.pt")
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        return fpath

    def get_scenario(self, scene_id: str, **kwargs) -> Dict[str, Any]:
        scenario_fpath = self.get_scenario_fpath(scene_id)
        with open(scenario_fpath, 'rb') as f:
            scenario = pickle.load(f)
        # return scenario
        return self._get_scenario(scenario, **kwargs)

    def get_scenario_background_only(self, scene_id: str, **kwargs):
        return self.get_scenario(scene_id, no_objects=True, **kwargs)

    def get_metadata(self, scene_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = os.path.join(self.root, scene_id, self.rgb_dirname, camera_id, idx_to_img_filename(frame_index))
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        # [H, W, 3]
        return load_rgb(fpath)

    def get_mono_depth(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = os.path.join(self.root, scene_id, self.rgb_mono_depth_dirname, camera_id, f'{idx_to_frame_str(frame_index)}.npz')
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        depth = np.load(fpath)['arr_0'].astype(np.float32)
        return depth

    def get_mono_normals(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = os.path.join(self.root, scene_id, self.rgb_mono_normals_dirname, camera_id, f'{idx_to_frame_str(frame_index)}.jpg')
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
    def get_occupancy_mask(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True) -> np.ndarray:
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.ones_like(raw).astype(np.bool8)
        ret[raw==cityscapes_classes_ind_map['sky']] = False
        # [H, W] 
        # Binary occupancy mask on RGB image. 1 for occpied, 0 for not.
        return ret.squeeze()
    def get_dynamic_mask(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True):
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.zeros_like(raw).astype(np.bool8)
        for cls in cityscapes_dynamic_classes:
            ind = cityscapes_classes_ind_map[cls]
            ret[raw==ind] = True
        # [H, W] 
        # Binary dynamic mask on RGB image. 1 for dynamic object, 0 for static.
        return ret.squeeze()
    def get_human_mask(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True):
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.zeros_like(raw).astype(np.bool8)
        for cls in cityscapes_human_classes:
            ind = cityscapes_classes_ind_map[cls]
            ret[raw==ind] = True
        # [H, W] 
        # Binary dynamic mask on RGB image. 1 for human-related object, 0 for other.
        return ret.squeeze()
    def get_road_mask(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True):
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.zeros_like(raw).astype(np.bool8)
        ret[raw==cityscapes_classes_ind_map['road']] = True
        # [H, W] 
        # Binary dynamic mask on RGB image. 1 for road semantics, 0 for other.
        return ret.squeeze()
    def get_semantic_mask_all(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True) -> np.ndarray:
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.full(raw.shape, -1, dtype=np.int16)
        for waymo_ind, dataset_class_str in enumerate(WAYMO_CLASSES):
            for cls in waymo_classes_in_cityscapes[dataset_class_str]:
                ind = cityscapes_classes_ind_map[cls]
                ret[raw==ind] = waymo_ind
        # Integer semantic mask on RGB image.
        return ret.squeeze()
    def get_semantic_mask_of_class(self, scene_id: str, camera_id: str, frame_index: int, dataset_class_str: str, *, compress=True) -> np.ndarray:
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.zeros_like(raw).astype(np.bool8)
        for cls in waymo_classes_in_cityscapes[dataset_class_str]:
            ind = cityscapes_classes_ind_map[cls]
            ret[raw==ind] = True
        # [H, W] 
        # Binary dynamic mask on RGB image. 1 for selected class.
        return ret.squeeze()
    # def get_mask_from_raw(self, raw: torch.Tensor, nonoccupied_class='sky'):
    #     return (raw != cityscapes_classes_ind_map[nonoccupied_class]).float()
    # def get_dynamic_mask_from_raw(self, raw: torch.Tensor):
    #     ret = raw.new_zeros(raw.shape, dtype=torch.bool)
    #     for cls in cityscapes_dynamic_classes:
    #         ind = cityscapes_classes_ind_map[cls]
    #         ret[raw==ind] = True
    #     return ret.float()
    # def get_semantic_mask_of_class_from_raw(self, raw: torch.Tensor, dataset_class_str: str):
    #     ret = raw.new_zeros(raw.shape, dtype=torch.bool)
    #     for cls in waymo_classes_in_cityscapes[dataset_class_str]:
    #         ind = cityscapes_classes_ind_map[cls]
    #         ret[raw==ind] = True
    #     return ret.float()

    def get_lidar(self, scene_id: str, lidar_id: str, frame_index: int) -> Dict[str, np.ndarray]:
        fpath = os.path.join(self.root, scene_id, self.lidar_dirname, lidar_id, idx_to_lidar_filename(frame_index))
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        arr_dict = np.load(fpath)
        # TODO: There can be a more elegant design here:
        #       When it's not the TOP lidar, we can store only azimuth and inclinations,
        #       and then calculate them before returning in the get function.
        return dict(rays_o=arr_dict['rays_o'], rays_d=arr_dict['rays_d'], ranges=arr_dict['ranges'])

    def _get_scenario(
        self, scenario: dict, # The original scenario loaded from preprocessed dataset
        *, 
        observer_cfgs: dict, # scene_bank's per class_name observer configs
        object_cfgs: dict = {}, # scene_bank's per class_name object configs
        no_objects=False, # Set to [true] to load no object at all.
        joint_camlidar=True, # Joint all cameras and lidars; Set this to [true] if you needs to calibrate cam/lidar extr
        joint_camlidar_equivalent_extr=True, # Set to [false] if you needs to calibrate cam/lidar extr
        consider_distortion=True, # Set to [true] to take care of camera distortions.
        align_orientation=True, # Set to [true] to rotate the street obj to align with major vehicle moving direction
        aabb_extend: float = 60.0, # Not actually used for now.
        start=None, stop=None, # (Optionally) Drop beggining frames or ending frames that we do not want.
        ):

        new_scenario = dict(scene_id=scenario['scene_id'], metas=scenario['metas'])
        data_frame_offset = new_scenario['metas']['data_frame_offset'] = start if start is not None else 0
        original_num_frames = scenario['metas']['n_frames']
        new_scenario['metas']['n_frames'] = (stop if stop is not None else original_num_frames) - data_frame_offset
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
        
        for oid, odict in scenario['objects'].items():
            o_class_name = odict['class_name']
            
            #---- Scene meta data for objects (bbox list of each frame)
            obj_box_list_per_frame.setdefault(o_class_name, [[] for _ in range(original_num_frames)])
            obj_box_list_per_frame_dynamic_only.setdefault(o_class_name, [[] for _ in range(original_num_frames)])
            for seg in odict['segments']:
                for seg_local_fi in range(seg['n_frames']):
                    fi = seg['start_frame'] + seg_local_fi
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
        frame_pose = scenario['observers']['ego_car']['data']['v2w']
        if joint_camlidar:
            ego_car = dict(
                class_name='EgoVehicle', 
                children=dict(), 
                n_frames=scenario['observers']['ego_car']['n_frames'], 
                data=dict(
                    transform=frame_pose, 
                    timestamp=scenario['observers']['ego_car']['data']['timestamp'], 
                    global_frame_ind=scenario['observers']['ego_car']['data']['global_frame_ind']
                )
            )
            ego_car = clip_node_data(ego_car, start, stop)
            new_scene_observers['ego_car'] = ego_car
        
        cam_intrs_all = []
        cam_c2ws_all = []
        cam_front_c2ws_all = []
        
        #------------------------------------------------------
        #------------------     Cameras      ------------------
        #------------------------------------------------------
        for oid, odict in scenario['observers'].items():
            if (o_class_name:=odict['class_name']) == 'Camera':
                hw = odict['data']['hw']
                intr = odict['data']['intr']
                distortion = odict['data']['distortion']
                c2w = odict['data']['c2w']
                c2v = odict['data']['c2v']
                v2w = odict['data']['sensor_v2w']
                timestamp = odict['data']['timestamp']
                global_frame_ind = odict['data']['global_frame_ind']
                
                cam_intrs_all.append(intr)
                cam_c2ws_all.append(c2w)
                
                if oid == 'camera_FRONT':
                    cam_front_c2ws_all.append(c2w)
                
                if o_class_name not in observer_cfgs.keys():
                    # Ignore un-wanted class_names
                    continue
                if oid not in observer_cfgs[o_class_name].list:
                    # Ignore un-wanted oberver id
                    continue
                
                camera_model = 'opencv' if consider_distortion else 'pinhole'
                if not joint_camlidar:
                    new_odict = dict(
                        class_name='Camera', n_frames=odict['n_frames'], 
                        camera_model=camera_model, 
                        data=dict(
                            timestamp=timestamp, global_frame_ind=global_frame_ind, 
                            hw=hw, intr=intr, distortion=distortion, 
                            transform=c2w
                        )
                    )
                    new_scene_observers[oid] = clip_node_data(new_odict, start, stop)
                else:
                    dpose = np.linalg.inv(frame_pose @ c2v) @ (v2w @ c2v)
                    if joint_camlidar_equivalent_extr:
                        new_odict = dict(
                            class_name='Camera', n_frames=odict['n_frames'], 
                            camera_model=camera_model, 
                            data=dict(
                                timestamp=timestamp, global_frame_ind=global_frame_ind, 
                                hw=hw, intr=intr, distortion=distortion, 
                                transform=c2v @ dpose
                                # transform=c2v, 
                                # dpose=dpose,
                            )
                        )
                        ego_car['children'][oid] = clip_node_data(new_odict, start, stop)
                    else: # TODO: Needed when calibrating cameras
                        raise NotImplementedError("jianfei: Not recommended until timestamp indexing is supported.")
                        new_odict = dict(
                            class_name='Camera', n_frames=odict['n_frames'], 
                            camera_model=camera_model, 
                            data=dict(
                                timestamp=timestamp, global_frame_ind=global_frame_ind, 
                                hw=hw, intr=intr, distortion=distortion, 
                                transform=c2v, dpose=dpose
                            )
                        )
                        ego_car['children'][oid] = clip_node_data(new_odict, start, stop)

        #------------------------------------------------------
        #-------------------     Lidars      ------------------
        #------------------------------------------------------
        for oid, odict in scenario['observers'].items():
            if (o_class_name:=odict['class_name']) == 'RaysLidar':
                l2v = odict['data']['l2v']
                timestamp = odict['data']['timestamp']
                global_frame_ind = odict['data']['global_frame_ind']
                
                if o_class_name not in observer_cfgs.keys():
                    # Ignore un-wanted class_names
                    continue
                if oid not in observer_cfgs[o_class_name].list:
                    # Ignore un-wanted oberver id
                    continue

                if joint_camlidar:
                    new_odict = dict(
                        class_name='RaysLidar', n_frames=odict['n_frames'], 
                        data=dict(timestamp=timestamp, global_frame_ind=global_frame_ind, 
                            transform=l2v
                        )
                    )
                    ego_car['children'][oid] = clip_node_data(new_odict, start, stop)
                else:
                    new_odict = dict(
                        class_name='RaysLidar', n_frames=odict['n_frames'], 
                        data=dict(timestamp=timestamp, global_frame_ind=global_frame_ind, 
                            transform=frame_pose @ l2v
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
        
        """
        < waymo convention >
        facing [+x] direction, z upwards, y left
                z ↑ 
                  |  ↗ x
                  | /
                  |/
         ←--------o
        y
        """
        # NOTE: Convert original [camera<openCV> to world<waymo>] to [camera<waymo> to world<waymo>] 
        #       (i.e. to get camera's poses in waymo)
        opencv_to_waymo = np.eye(4)
        opencv_to_waymo[:3 ,:3] = np.array([[0, 0, 1],[-1, 0, 0],[0, -1, 0]])
        waymo_to_opencv = np.linalg.inv(opencv_to_waymo)
        cam_front_poses_in_waymo = cam_front_c2ws_all @ waymo_to_opencv[None, ...]
        cam_front_tracks_in_waymo = cam_front_poses_in_waymo[:, :3, 3]
        
        # NOTE: Average rotation about z axis.
        #---- Option1: discarded in the beginning.
        # r = R.from_matrix(cam_front_poses_in_waymo[:, :3, :3])
        # rot_zyx = r.as_euler('zyx', degrees=False)
        # avg_rot_z = rot_zyx[:, 0].mean()
        
        #---- Option 2: 22Q4
        r = R.from_matrix(cam_front_poses_in_waymo[:, :3, :3])
        avg_rot = r.mean()
        avg_rot_zyx = avg_rot.as_euler('zyx', degrees=False)
        avg_rot_z: float = avg_rot_zyx[0]
        avg_rot_z_mat = R.from_rotvec(np.array([0, 0, avg_rot_z])).as_matrix()
        
        #---- Option 3: 23Q1 new: Since only rot-z angle is needed
        # cam_front_tracks_xy_in_waymo = cam_front_poses_in_waymo[:, :2, 3]
        # # Option 3.1: The secant direction from start point to stop point of the track
        # delta_xy = cam_front_tracks_xy_in_waymo[-1] - cam_front_tracks_xy_in_waymo[0]
        # avg_rot_z: float = np.arctan2(delta_xy[1], delta_xy[0])
        # # Option 3.2: The average tangent direction of the track # NOTE: might suffer from numerical error since some delta can be very small
        # # delta_xys = (cam_front_tracks_xy_in_waymo[1:] - cam_front_tracks_xy_in_waymo[:-1])
        # # avg_rot_zs: float = np.arctan2(delta_xys[:, 1], delta_xys[:, 0])
        # # avg_rot_z: float = avg_rot_zs.mean()
        
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
        new_scenario['metas']['align_orientation'] = align_orientation
        
        aabb_min = cam_front_tracks_in_waymo.min(axis=0) - aabb_extend
        aabb_max = cam_front_tracks_in_waymo.max(axis=0) + aabb_extend
        new_scenario['metas']['aabb'] = np.stack([aabb_min, aabb_max], axis=0)

        return new_scenario

if __name__ == "__main__":
    def make_unit_test_dataloader(device, preload=False):
        from nr3d_lib.utils import import_str
        from nr3d_lib.config import ConfigDict
        from app.resources import create_scene_bank
        from dataio.dataloader import SceneDataLoader
        from app.resources.asset_bank import AssetBank
        dataset_cfg = ConfigDict(
            target='dataio.autonomous_driving.WaymoDataset', 
            param=ConfigDict(
                root='/data1/waymo/processed', 
                rgb_dirname="images", 
                lidar_dirname="lidars", 
                # lidar_dirname="lidars_ds=4", 
                mask_dirname="masks", 
            )
        )
        scenebank_cfg = ConfigDict(
            scenarios=['segment-7670103006580549715_360_000_380_000_with_camera_labels, 15'], 
            # scenarios=['segment-16646360389507147817_3320_000_3340_000_with_camera_labels'], 
            # scenarios=['segment-10061305430875486848_1080_000_1100_000_with_camera_labels, 0, 163'], 
            observer_cfgs=ConfigDict(
                Camera=ConfigDict(
                    list=['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT', 'camera_SIDE_LEFT', 'camera_SIDE_RIGHT']
                ), 
                RaysLidar=ConfigDict(
                    list=['lidar_TOP', 'lidar_FRONT', 'lidar_REAR', 'lidar_SIDE_LEFT', 'lidar_SIDE_RIGHT']
                    # list=['lidar_TOP']
                    # list=['lidar_FRONT', 'lidar_REAR', 'lidar_SIDE_LEFT', 'lidar_SIDE_RIGHT']
                ), 
            ), 
            object_cfgs=ConfigDict(
                Vehicle=ConfigDict(
                    dynamic_only=True
                ),
                Pedestrian=ConfigDict(
                    dynamic_only=True
                )
            ), 
            no_objects=False, 
            align_orientation=False, 
            aabb_extend=120., 
            consider_distortion=True, 
            joint_camlidar=True,
            joint_camlidar_equivalent_extr=True, 
        )
        assetbank_cfg = ConfigDict({
            'Vehicle': {'model_class': 'app.models.base.AD_DummyBox', 'model_params': {}}, 
            'Pedestrian': {'model_class': 'app.models.base.AD_DummyBox', 'model_params': {}}, 
            'Street': {'model_class': 'app.models.base.DummyBox', 'model_params': {}}, 
            # 'Distant': {}
        })
        
        dataset_impl: DatasetIO = import_str(dataset_cfg.target)(dataset_cfg.param)
        scene_bank, scene_bank_meta = create_scene_bank(
            dataset=dataset_impl, device=device, 
            scenebank_root=None,
            scenebank_cfg=scenebank_cfg, 
            drawable_class_names=assetbank_cfg.keys(), 
            misc_node_class_names=['node', 'EgoVehicle', 'EgoDrone'], 
        )
        scene = scene_bank[0]
        # scene.debug_vis_multi_frame(40)
        scene_dataloader = SceneDataLoader(
            scene_bank, dataset_impl, device=device, 
            config=ConfigDict(
                preload=preload, 
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
                            # filter_in_cams=True, 
                            filter_out_objs=False
                        )
                    )
                )
            ))
        asset_bank = AssetBank(assetbank_cfg)
        asset_bank.create_asset_bank(scene_bank, load_assets_into_scene=True)
        scene.load_assets(asset_bank)
        return scene_bank, scene_dataloader

    def test_scenegraph():
        import torch
        from icecream import ic
        device = torch.device('cuda')
        scene_bank, scene_dataloader = make_unit_test_dataloader(device, preload=False)
        scene = scene_bank[0]
        # scene.debug_vis_scene_graph(120, arrow_length=0.5, font_scale=1.0)
        scene.debug_vis_scene_graph(120)

    def test_scenario():
        import torch
        from icecream import ic
        device = torch.device('cuda')
        scene_bank, scene_dataloader = make_unit_test_dataloader(device, preload=False)
        scene = scene_bank[0]
        mesh_file = '/data2/lidar_only/exp1_lidaronly_5lidar_filterobj_dynamic_ext80.0_7.5k_all_wli=0.03/seg7670103/meshes/seg7670103_exp1_wli0.03_res=0.1.ply'
        scene.debug_vis_anim(
            scene_dataloader=scene_dataloader, 
            #  plot_image=True, camera_length=8., 
            plot_lidar=True, lidar_pts_ds=2, 
            # mesh_file=mesh_file
        )

    def test_lidar():
        import torch
        from nr3d_lib.plot import vis_lidar_vedo
        from app.resources.observers.lidars import MultiRaysLidarBundle
        device = torch.device('cuda')
        scene_bank, scene_dataloader = make_unit_test_dataloader(device, preload=False)
        scene = scene_bank[0]
        
        frame_ind = 41
        lidar_gts = scene_dataloader.get_merged_lidar_gts(scene.id, frame_ind, device=device, filter_if_configured=True)
        scene.frozen_at(frame_ind)
        lidars = scene.all_nodes_by_class_name['RaysLidar'][scene_dataloader.lidar_id_list]
        lidars = MultiRaysLidarBundle(lidars)
        l2w = lidars.world_transform[lidar_gts['i']]
        pts_local = torch.addcmul(lidar_gts['rays_o'], lidar_gts['rays_d'], lidar_gts['ranges'].unsqueeze(-1))
        pts = l2w.forward(pts_local)
        pts = pts.data.cpu().numpy()
        # vis_lidar_o3d(pts)
        vis_lidar_vedo(pts)

    def test_distort():
        import torch
        import imageio
        from nr3d_lib.utils import check_to_torch, cond_mkdir
        from kornia.geometry.calibration.undistort import undistort_image
        device=torch.device('cuda')
        config = ConfigDict(
            root='/data1/waymo/processed/', 
            rgb_dirname='images', 
        )
        dataset = WaymoDataset(config)
        
        scene_id = 'segment-7670103006580549715_360_000_380_000_with_camera_labels'
        cam_id = 'camera_FRONT'
        # frame_ind = 83
        frame_ind = 159
        scenario = dataset.get_scenario(scene_id)
        odict = scenario['observers']['ego_car']['children'][cam_id]
        K = check_to_torch(odict['data']['intr'][frame_ind], device=device, dtype=torch.float)
        dist = check_to_torch(odict['data']['distortion'][frame_ind], device=device, dtype=torch.float)
        rgb = dataset.get_image(scene_id, cam_id, frame_ind)
        # [H, W, C] -> [C, H, W]
        rgb_tensor = check_to_torch(rgb, device=device, dtype=torch.float).movedim(-1, 0)
        rgb0_tensor = undistort_image(rgb_tensor, K, dist)
        # {C, H, W} -> [H, W, C]
        rgb0 = rgb0_tensor.movedim(0, -1).contiguous().data.cpu().numpy()
        cond_mkdir('./dev_test/test_distortion')
        imageio.imwrite(f'./dev_test/test_distortion/frame_{frame_ind}.png', rgb)
        imageio.imwrite(f'./dev_test/test_distortion/frame_{frame_ind}_undist.png', rgb0)

    def test_mask():
        import torch
        import kornia
        from nr3d_lib.plot import vis_lidar_vedo
        from app.resources.observers.lidars import MultiRaysLidarBundle
        device = torch.device('cuda')
        scene_bank, scene_bank_loader = make_unit_test_dataloader(device, preload=False)
        scene = scene_bank[0]
        
        mask = scene_bank_loader.get_occupancy_mask(scene.id, 'camera_FRONT', 45, device=device)
        gt = scene_bank_loader.get_rgb(scene.id, 'camera_FRONT', 45, device=device)['rgb']
        
        from nr3d_lib.utils import img_to_torch_and_downscale
        mask = img_to_torch_and_downscale(mask, downscale=2)
        gt = img_to_torch_and_downscale(gt, downscale=2)
        
        import matplotlib.pyplot as plt
        mask_erode = 20 # 10 pixels of erosion
        mask_new = kornia.morphology.erosion(mask[None,None].float(), torch.ones([mask_erode,mask_erode], device=mask.device))[0,0].bool()
        mask = mask.data.cpu().numpy()
        mask_new = mask_new.data.cpu().numpy()
        im = np.zeros_like(mask, dtype=float)
        im[mask_new] += 0.4
        im[mask] += 0.6
        plt.imshow(im, vmin=0., vmax=1.)
        plt.show()

    # test_scenegraph()
    test_scenario()
    # test_lidar()
    # test_mask()
    # test_distort()