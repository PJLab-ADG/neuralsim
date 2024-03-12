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
from typing import Any, Dict, List, Literal, Tuple, Union
from scipy.spatial.transform import Rotation as R

from nr3d_lib.utils import get_image_size, load_rgb
from nr3d_lib.config import ConfigDict
from nr3d_lib.fmt import log

from dataio.scene_dataset import SceneDataset
from dataio.utils import clip_node_data, clip_node_segments

#---------------- Waymo original definition
WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
WAYMO_CAMERAS = ['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT'] # NOTE: name order in frame.images
WAYMO_LIDARS = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR'] # NOTE: name order in frame.lasers
# NUM_CAMERAS = len(WAYMO_CAMERAS)
# NUM_LIDARS = len(WAYMO_LIDARS)

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
class WaymoDataset(SceneDataset):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.populate(**config)

    def populate(
        self, root: str, 
        rgb_dirname: str = "images", 
        lidar_dirname: str = "lidars", 
        mask_dirname: str = "masks", 
        mask_taxonomy: Literal['cityscapes', 'ade20k'] = 'cityscapes', 
        image_mono_depth_dirname: str = "depths", 
        image_mono_normals_dirname: str = "normals", 
        pcl_dirname: str = None, 
        ):

        assert os.path.exists(root), f"Not exist: {root}"

        self.main_class_name = "Street"
        self.root = root
        
        self.rgb_dirname = rgb_dirname
        self.image_mono_depth_dirname = image_mono_depth_dirname
        self.image_mono_normals_dirname = image_mono_normals_dirname
        self.lidar_dirname = lidar_dirname
        self.pcl_dirname = pcl_dirname
        
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

    def get_image_wh(self, scene_id: str, camera_id: str, frame_index: Union[int, List[int]]) -> np.ndarray:
        def get_single_frame_wh(fi: int):
            fpath = os.path.join(self.root, scene_id, self.rgb_dirname, camera_id, idx_to_img_filename(frame_index))
            assert os.path.exists(fpath), f"Not exist: {fpath}"
            return get_image_size(fpath)
        if isinstance(frame_index, int):
            WH = get_single_frame_wh(frame_index)
        elif isinstance(frame_index, list):
            WH = [get_single_frame_wh(fi) for fi in frame_index]
        else:
            raise RuntimeError(f"Invalid type(frame_index)={type(frame_index)}")
        WH = np.array(WH)
        return WH

    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = os.path.join(self.root, scene_id, self.rgb_dirname, camera_id, idx_to_img_filename(frame_index))
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        # [H, W, 3]
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
    
    def get_image_semantic_mask_all(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True) -> np.ndarray:
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.full(raw.shape, -1, dtype=np.int16)
        for waymo_ind, dataset_class_str in enumerate(WAYMO_CLASSES):
            for cls in self.dataset_classes_in_sematic[dataset_class_str]:
                ind = self.semantic_classes_ind_map[cls]
                ret[raw==ind] = waymo_ind
        # Integer semantic mask on RGB image.
        return ret.squeeze()
    def get_semantic_mask_of_class(self, scene_id: str, camera_id: str, frame_index: int, dataset_class_str: str, *, compress=True) -> np.ndarray:
        raw = self.get_raw_mask(scene_id, camera_id, frame_index, compress=compress)
        ret = np.zeros_like(raw).astype(np.bool8)
        for cls in self.dataset_classes_in_sematic[dataset_class_str]:
            ind = self.semantic_classes_ind_map[cls]
            ret[raw==ind] = True
        # [H, W] 
        # Binary dynamic mask on RGB image. 1 for selected class.
        return ret.squeeze()

    def get_lidar(self, scene_id: str, lidar_id: str, frame_index: int) -> Dict[str, np.ndarray]:
        fpath = os.path.join(self.root, scene_id, self.lidar_dirname, lidar_id, idx_to_lidar_filename(frame_index))
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        arr_dict = np.load(fpath)
        # TODO: There can be a more elegant design here:
        #       When it's not the TOP lidar, we can store only azimuth and inclinations,
        #       and then calculate them before returning in the get function.
        return dict(rays_o=arr_dict['rays_o'], rays_d=arr_dict['rays_d'], ranges=arr_dict['ranges'])

    def _get_scenario(
        self, 
        scenario: dict, *, 
        observer_cfgs: dict,
        object_cfgs: dict = {},
        no_objects=False,
        load_class_names: List[str]=['Street'], 
        consider_distortion=True,
        scene_graph_has_ego_car=True,
        align_orientation=True,
        normalize_ts=True, 
        use_ts_interp=False, 
        how_to_account_for_cam_timestamp_diff: \
            Literal['use_ts_interp', 'correct_extrinsics'] \
            = 'correct_extrinsics',  
        
        aabb_extend: float = 60.0,
        start=None, stop=None,
        
        # DEPRECATED backups
        joint_camlidar=None, 
        joint_camlidar_equivalent_extr=None, 
        correct_extr_for_timestamp_difference=None,
        ) -> dict:
        """ Convert preprocessed `scenario.pt` to a dict loadable by `Scene` structure, 
            with some configurable dataset-specific functionalities.
        
        Args:
            scenario (dict): The raw scenario info loaded from preprocessed pickle
            observer_cfgs (dict): Observer loading config for each class_name
            object_cfgs (dict, optional): Other objects loading config for each class_name. Defaults to {}.
            no_objects (bool, optional): If true, will skip loading of all other objects (only preserve ego). Defaults to False.
            consider_distortion (bool, optional): If true, will respect the distortion calibration. Defaults to True.
            scene_graph_has_ego_car (bool, optional): Whether contain ego_car in scene_graph. Defaults to True.
                If true, the scene_graph will be: world -> ego_car -> camera,lidar,...\
                    In this case, the transform of sensors is their pose in ego_car
                If false, the scene_graph will be: world -> camera,lidar,...\
                    In this case, the transform of sensors is their pose in world
            how_to_account_for_cam_timestamp_diff (str, optional): Specify how do we account for timestamp difference of cameras.
                - `use_ts_interp`: The scene nodes pose will be interpolated on continuous timestamps. \
                - `correct_extrinsics`: Modify the camera-to-vehicle transform at each frame \
                    with the pose changes between the capture time of the current camera \
                    and the capture time of the frame / ego_car (for waymo, it's same with camera_FRONT).
            use_ts_interp (bool, optional): Whether the scene is using timestamp indexing. \
                Defaults to True.
            align_orientation (bool, optional): If true, will align the `street` objects rotation about z-axis \
                with the average orientation of the camera trajectory. \
                Defaults to True.
            aabb_extend (float, optional): Currently not used. Defaults to 60.0.
            start (int, optional): An optional starting index to trim the data sequence. Defaults to None.
            stop (int, optional): An optional ending index to trim the data sequence. Defaults to None.

        Returns:
            dict: The scenario dict directly loadable by Scene structure.
        """        
        
        #---- NOTE: Deprecation warning
        if joint_camlidar is not None:
            log.warning("neuralsim: `joint_camlidar` is DEPRECATED. "\
                "Please rename to `scene_graph_has_ego_car`. "\
                "Check docstring for more details.")
            scene_graph_has_ego_car = joint_camlidar
        if joint_camlidar_equivalent_extr is not None:
            # log.warning("neuralsim: `joint_camlidar_equivalent_extr` is DEPRECATED. Please rename to `correct_extr_for_timestamp_difference`. Check docstring for more details.")
            # correct_extr_for_timestamp_difference = joint_camlidar_equivalent_extr
            log.warning("neuralsim: `joint_camlidar_equivalent_extr` is DEPRECATED. "\
                "Please use `how_to_account_for_cam_timestamp_diff=correct_extrinsics` instead. "\
                "Check docstring for more details.")
            how_to_account_for_cam_timestamp_diff = 'correct_extrinsics'
        if correct_extr_for_timestamp_difference is not None:
            log.warning("neuralsim: `correct_extr_for_timestamp_difference` is DEPRECATED. "\
                "Please use `how_to_account_for_cam_timestamp_diff=correct_extrinsics` instead. "\
                "Check docstring for more details.")
            how_to_account_for_cam_timestamp_diff = 'correct_extrinsics'
        
        if how_to_account_for_cam_timestamp_diff == 'use_ts_interp':
            assert use_ts_interp, \
                f"how_to_account_for_cam_timestamp_diff={how_to_account_for_cam_timestamp_diff}"\
                "is only supported when use_ts_interp=True"
        
        original_n_frames = scenario['metas']['n_frames']
        original_ts = scenario['metas']['frame_timestamps']
        data_frame_offset = start if start is not None else 0
        new_n_frames = (stop if stop is not None else original_n_frames) - data_frame_offset
        
        # Normalize timestamps
        if normalize_ts:
            original_dt = (original_ts[-1] - original_ts[0]) / (len(original_ts)-1)
            new_dt = 2.0 / (len(original_ts)-1)
            ts_scale = 0.95 * (new_dt / original_dt) # A safe factor 0.95
            ts_offset = original_ts[len(original_ts)//2]
        else:
            ts_scale = 1.0
            ts_offset = 0.0
        new_ts = original_ts[data_frame_offset:data_frame_offset+new_n_frames]
        new_ts = (new_ts - ts_offset) * ts_scale
        
        # Allocate new scenario
        new_scenario = dict(scene_id=scenario['scene_id'], metas=scenario['metas'])
        new_scenario['metas']['use_ts_interp'] = use_ts_interp
        
        new_scenario['metas']['n_frames'] = new_n_frames
        new_scenario['metas']['frame_timestamps'] = new_ts

        new_scenario['metas']['data_frame_offset'] = data_frame_offset
        new_scenario['metas']['data_timestamp_scale'] = ts_scale
        new_scenario['metas']['data_timestamp_offset'] = ts_offset
        
        new_scene_objects = new_scenario['objects'] = dict()
        
        #------------------------------------------------------
        #------------------     Street      -------------------
        #------------------------------------------------------
        # The major class_name that we focus on in street-view surface reconstruction task
        new_scenario['metas']['main_class_name'] = self.main_class_name
        if self.main_class_name in load_class_names:
            street_odict = dict(id=f"street", class_name=self.main_class_name)
            new_scene_objects[street_odict['id']] = street_odict
        if 'Dynamic' in load_class_names:
            dynamic_odict = dict(id=f"dynamic", class_name="Dynamic")
            new_scene_objects[dynamic_odict['id']] = dynamic_odict
        
        #------------------------------------------------------
        #------------------     Objects      ------------------
        #------------------------------------------------------
        obj_box_list_per_frame = dict()
        obj_box_list_per_frame_dynamic_only = dict()
        
        for oid, odict in scenario['objects'].items():
            o_class_name = odict['class_name']
            
            #---- Scene meta data for objects (bbox list of each frame)
            obj_box_list_per_frame.setdefault(o_class_name, [[] for _ in range(original_n_frames)])
            obj_box_list_per_frame_dynamic_only.setdefault(o_class_name, [[] for _ in range(original_n_frames)])
            for seg in odict['segments']:
                if 'data' in seg and 'global_timestamps' in seg['data']:
                    seg['data']['global_timestamps'] = (seg['data']['global_timestamps'] - ts_offset) * ts_scale
                for seg_local_fi in range(seg['n_frames']):
                    fi = seg['start_frame'] + seg_local_fi
                    # transform_in_world (12) + scale (3)
                    cur_box = np.concatenate([seg['data']['transform'][seg_local_fi][:3, :].reshape(-1), seg['data']['scale'][seg_local_fi]])
                    obj_box_list_per_frame[o_class_name][fi].append(cur_box)
                    if oid in scenario['metas']['dynamic_stats'][o_class_name]['is_dynamic']:
                        obj_box_list_per_frame_dynamic_only[o_class_name][fi].append(cur_box)
            
            if no_objects or (o_class_name not in load_class_names):
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
        frame_timestamp = scenario['observers']['ego_car']['data']['global_timestamps']
        frame_timestamp = (frame_timestamp - ts_offset) * ts_scale
        if scene_graph_has_ego_car:
            ego_car = dict(
                class_name='EgoVehicle', 
                children=dict(), 
            )
            if use_ts_interp and how_to_account_for_cam_timestamp_diff == 'use_ts_interp':
                # NOTE: Concat ego pose at all 5 cam's timestamp
                #       In this case, attr data of `ego_car` will have 5x density compared to other nodes'
                #       i.e. if other nodes have 200 frames, ego_car will have 1000 frames.
                lst_all_cam_data_dict = [odict['data'] for odict in scenario['observers'].values() \
                    if odict['class_name']=='Camera']
                all_cam_global_ts = np.concatenate([d['global_timestamps'] for d in lst_all_cam_data_dict], axis=0)
                all_cam_global_ts = (all_cam_global_ts - ts_offset) * ts_scale
                all_cam_global_fi = np.concatenate([d['global_frame_inds'] for d in lst_all_cam_data_dict], axis=0)
                all_cam_ego_car_v2w = np.concatenate([d['sensor_v2w'] for d in lst_all_cam_data_dict], axis=0)
                local_fi_to_sort = np.argsort(all_cam_global_ts, axis=0)

                ego_car['n_frames'] = len(local_fi_to_sort)
                ego_car['data'] = dict(
                    transform=all_cam_ego_car_v2w[local_fi_to_sort], 
                    global_timestamps=all_cam_global_ts[local_fi_to_sort], 
                    global_frame_inds=all_cam_global_fi[local_fi_to_sort], 
                )
            else:
                ego_car['n_frames'] = scenario['observers']['ego_car']['n_frames']
                ego_car['data'] = dict(
                    transform=frame_pose, 
                    global_timestamps=frame_timestamp, 
                    global_frame_inds=scenario['observers']['ego_car']['data']['global_frame_inds'], 
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
                v2w = odict['data']['sensor_v2w'] # v2w at each camera's timestamp
                cam_timestamp = odict['data']['global_timestamps']
                cam_timestamp = (cam_timestamp - ts_offset) * ts_scale
                global_frame_inds = odict['data']['global_frame_inds']
                
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
                if scene_graph_has_ego_car:
                    new_odict = dict(
                        class_name='Camera', 
                        n_frames=odict['n_frames'], 
                        camera_model=camera_model, 
                        data=dict(
                            hw=hw, intr=intr, distortion=distortion, 
                            global_frame_inds=global_frame_inds
                        )
                    )

                    """
                    NOTE: Explanation for `dpose`, `correct_extr_for_timestamp_difference`:
                    Even at the same frame index, different camera footages are captured in different timestamps \
                        and thus with different ego_car's pose.
                    Currently, we use frame index only as the time axes, and do not support timestamp indexing.
                    Hence, to still allow for accurate camera pose, we correct the camera-to-vehicle transform \
                        at each frame with the ego_car's pose changes between the capture time of the current camera \
                            and the capture time of the reference camera (for waymo, it's camera_FRONT).
                    """
                    dpose = np.linalg.inv(frame_pose @ c2v) @ (v2w @ c2v)
                    """
                    NOTE: Even in frame index mode, not timestamp mode, different cameras having different timestamps still makes sense, as they indeed observe at different times.
                    """
                    if use_ts_interp:
                        if how_to_account_for_cam_timestamp_diff == 'correct_extrinsics':
                            new_odict['data'].update(global_timestamps=frame_timestamp, transform=c2v @ dpose)
                        elif how_to_account_for_cam_timestamp_diff == 'use_ts_interp':
                            new_odict['data'].update(global_timestamps=cam_timestamp, transform=c2v, dpose=dpose)
                        else:
                            raise RuntimeError("Invalid `how_to_account_for_cam_timestamp_diff`"\
                                f"={how_to_account_for_cam_timestamp_diff}")
                    else:
                        assert how_to_account_for_cam_timestamp_diff == 'correct_extrinsics'
                        new_odict['data'].update(global_timestamps=cam_timestamp, transform=c2v @ dpose, dpose=dpose)
                    ego_car['children'][oid] = clip_node_data(new_odict, start, stop)
                else:
                    new_odict = dict(
                        class_name='Camera', n_frames=odict['n_frames'], 
                        camera_model=camera_model, 
                        data=dict(
                            global_timestamps=cam_timestamp, 
                            global_frame_inds=global_frame_inds, 
                            hw=hw, intr=intr, distortion=distortion, 
                            transform=c2w
                        )
                    )
                    new_scene_observers[oid] = clip_node_data(new_odict, start, stop)

        #------------------------------------------------------
        #-------------------     Lidars      ------------------
        #------------------------------------------------------
        for oid, odict in scenario['observers'].items():
            if (o_class_name:=odict['class_name']) == 'RaysLidar':
                l2v = odict['data']['l2v']
                lidar_timestamp = odict['data']['global_timestamps']
                lidar_timestamp = (lidar_timestamp - ts_offset) * ts_scale
                
                if o_class_name not in observer_cfgs.keys():
                    # Ignore un-wanted class_names
                    continue
                if oid not in observer_cfgs[o_class_name].list:
                    # Ignore un-wanted oberver id
                    continue

                if scene_graph_has_ego_car:
                    new_odict = dict(
                        class_name='RaysLidar', n_frames=odict['n_frames'], 
                        data=dict(
                            global_timestamps=lidar_timestamp, 
                            global_frame_inds=odict['data']['global_frame_inds'], 
                            transform=l2v
                        )
                    )
                    ego_car['children'][oid] = clip_node_data(new_odict, start, stop)
                else:
                    new_odict = dict(
                        class_name='RaysLidar', n_frames=odict['n_frames'], 
                        data=dict(
                            global_timestamps=lidar_timestamp, 
                            global_frame_inds=odict['data']['global_frame_inds'], 
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

