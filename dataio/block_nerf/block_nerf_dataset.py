"""
@file   block_nerf_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset IO for Waymo's BlockNeRF dataset.

Modified from https://github.com/dvlab-research/LargeScaleNeRFPytorch
We directly use their already pre-processed data.
"""
import os
import json
import itertools
import numpy as np
from typing import Any, Dict, List, Literal, Union

import torch

from nr3d_lib.utils import load_rgb
from nr3d_lib.config import ConfigDict

from dataio.scene_dataset import SceneDataset

class BlockNeRFDataset(SceneDataset):
    def __init__(self, config:ConfigDict) -> None:
        self.config = config
        self.populate(**config)
    
    def populate(
        self, root_dir: str, split='train', 
        block_ids: str = None, cam_ids: str = None, exposures_std: float = 1.0, # Assume it's already in a normalized range.
        # exposures_std: float = 0.005307023699206318, # Calculated std()
        mask_dirname: str = "masks", 
        mask_taxonomy: Literal['cityscapes', 'ade20k'] = 'cityscapes', 
        ):
        
        self.main_class_name = "Street"
        self.split = split
        self.cam_ids = cam_ids
        self.block_ids = block_ids
        if self.block_ids is not None:
            if not isinstance(self.block_ids, list):
                self.block_ids = [str(self.block_ids)]
            else:
                self.block_ids = [str(i) for i in self.block_ids]
        if self.cam_ids is not None:
            if not isinstance(self.cam_ids, list):
                self.cam_ids = [str(self.cam_ids)]
            else:
                self.cam_ids = [str(i) for i in self.cam_ids]
        
        self.root_dir = root_dir
        self.meta = torch.load(os.path.join(self.root_dir, f'train/train_all_meta.pt'))
        if self.block_ids is None:
            img_names = list(self.meta.keys())
        else:
            with open(os.path.join(self.root_dir, f'train/split_block_train.json'), 'r') as fp:
                self.block_split_info: dict = json.load(fp)
            img_names = list(set(itertools.chain.from_iterable([list(zip(*self.block_split_info[bid]['elements']))[0] for bid in self.block_ids])))

        self.img_names: List[str] = []
        self.cam_img_names_map: Dict[str, List[str]] = {}
        self.cam_inds_map: Dict[str, np.ndarray] = {}
        self.c2ws_all: Dict[str, np.ndarray] = {}
        self.intrs_all: Dict[str, np.ndarray] = {}
        self.hws_all: Dict[str, np.ndarray] = {}
        self.exposures: Dict[str, float] = {}
        
        self._populate_mask_settings(mask_dirname=mask_dirname, mask_taxonomy=mask_taxonomy)
        
        # NOTE: Block-nerf-pytorch use OpenGL camera coordiantes
        """
            < opencv / colmap convention >                 --->>>   < openGL convention >
            facing [+z] direction, x right, y downwards    --->>>  facing [-z] direction, x right, y upwards, 
                        z                                                ↑ y               
                      ↗                                                  |                  
                     /                                                   |               
                    /                                                    |                 
                    o------> x                                           o------> x      
                    |                                                   /                       
                    |                                                  /                    
                    |                                                 ↙              
                    ↓                                               z                
                    y                                                                        
        """
        opencv_to_opengl = np.eye(4)
        opencv_to_opengl[:3, :3] = np.array(
            [[1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]]
        )

        i = 0
        for img_name in img_names:
            img_info = self.meta[img_name]
            assert img_name == img_info['image_name']
            cam_id = str(img_info['cam_idx'])
            if self.cam_ids is not None and cam_id not in self.cam_ids:
                continue
            
            self.img_names.append(img_name)
            self.cam_inds_map.setdefault(cam_id, []).append(i)
            i += 1
            self.cam_img_names_map.setdefault(cam_id, []).append(img_name)
            
            exposure = img_info['equivalent_exposure']
            c2w = np.eye(4)
            c2w[:3, :4] = img_info['c2w'].float().numpy()[:3, :4]
            c2w = c2w @ opencv_to_opengl
            
            if img_name in self.c2ws_all:
                raise Exception()
            self.c2ws_all[img_name] = c2w
            
            W, H = img_info['W'], img_info['H']
            K = np.zeros([3,3], dtype=np.float32)
            K[0, 0] = img_info['intrinsics'][0].item()
            K[1, 1] = img_info['intrinsics'][1].item()
            K[0, 2] = W * 0.5
            K[1, 2] = H * 0.5
            K[2, 2] = 1
            
            self.intrs_all[img_name] = K
            self.hws_all[img_name] = [H, W]
            self.exposures[img_name] = exposure

        self.hws_all = np.array(list(self.hws_all.values()))
        self.c2ws_all = np.stack(list(self.c2ws_all.values()), axis=0)
        self.intrs_all = np.stack(list(self.intrs_all.values()), axis=0)
        self.exposures = np.array(list(self.exposures.values())) / exposures_std
        self.cam_inds_map = {k: np.array(v) for k, v in self.cam_inds_map.items()}
        self.n_images = len(self.img_names)

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

    def get_scenario(self, scene_id: str, **kwargs) -> Dict[str, Any]:
        metas = dict(
            n_frames=self.n_images, 
            main_class_name=self.main_class_name
        )
        cam = dict(
            id='camera',
            class_name='Camera', 
            n_frames=self.n_images, 
            data=dict(
                hw=self.hws_all, 
                intr=self.intrs_all,
                transform=self.c2ws_all,
                exposure=self.exposures, 
                global_frame_inds=np.arange(self.n_images), 
            )
        )
        obj = dict(
            id=self.main_class_name.lower(),
            class_name=self.main_class_name, 
            # Has no recorded data.
        )
        scenario = dict(
            scene_id=f"waymo-block-nerf", 
            metas=metas, 
            objects={obj['id']: obj}, 
            observers={cam['id']: cam}
        )
        return scenario

    def get_image_wh(self, scene_id: str, camera_id: str, frame_index: Union[int, List[int]]):
        return self.hws_all[frame_index][::-1]

    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        img_name = self.img_names[frame_index]
        fpath = os.path.join(self.root_dir, self.split, 'rgbs', f"{img_name}.png")
        return load_rgb(fpath) 

    def get_image_mono_depth(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        img_name = self.img_names[frame_index]
        fpath = os.path.join(self.root_dir, self.split, 'depths', f'{img_name}.npz')
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        depth = np.load(fpath)['arr_0'].astype(np.float32)
        return depth

    def get_image_mono_normals(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        img_name = self.img_names[frame_index]
        fpath = os.path.join(self.root_dir, self.split, 'normals', f'{img_name}.jpg')
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        # [-1.,1.] np.float32
        normals = load_rgb(fpath)*2-1
        return normals

    def get_exposure(self, scene_id: str, camera_id: str, frame_index: int) -> float:
        img_name = self.img_names[frame_index]
        return self.exposures[img_name]

    def get_raw_mask(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        img_name = self.img_names[frame_index]
        fpath = os.path.join(self.root_dir, self.split, 'masks', f"{img_name}.npz")
        arr = np.load(fpath)['arr_0']
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

    def get_aabb(self, scene_id: str) -> Dict[str, np.ndarray]:
        raise NotImplementedError

if __name__ == "__main__":
    def unit_test():
        from icecream import ic
        dataset_cfg = ConfigDict(
            target="dataio.block_nerf.BlockNeRFDataset", 
            param=ConfigDict(
                root_dir="/data1/waymo/waymo-block-nerf/pytorch_waymo_dataset", 
                split="train", 
                block_ids=['block_0'], 
                cam_ids=[69]
            )
        )
        
        # dataset_cfg.cam_ids = [117]
        # dataset_cfg.block_ids = ['block_0', 'block_1']
        dataset = BlockNeRFDataset(dataset_cfg)
        
        # 44 blocks, ~500-800 `elements` each. (NOTE: there is overlap between blocks.)
        # lengths = [len(binfo['elements']) for bid, binfo in dataset.block_split_info.items()]

        cam_id = {v['cam_idx']:... for v in dataset.meta.values()}
        cam_id = sorted(list(cam_id.keys()))
        print(f"Cameras: {cam_id}")
        # [69, 71, 73, 75, 77, 79, 81, 83, 115, 117, 119, 121]
        
        # 0.0053070236 -> Use this value to normalize `exposures`
        ic(dataset.exposures.std())

        import matplotlib.pyplot as plt
        plt.hist(dataset.exposures)
        plt.show()
        
        for i, (cam_id, inds) in enumerate(dataset.cam_inds_map.items()):
            plt.subplot(3, 4, i+1)
            plt.hist(dataset.exposures[inds])
            plt.title(f"cam_idx={cam_id}")
        plt.show()


        c0 = np.array(dataset.block_split_info['block_0']['centroid'][1])
        c1 = np.array(dataset.block_split_info['block_1']['centroid'][1])
        c2 = np.array(dataset.block_split_info['block_2']['centroid'][1])
        c3 = np.array(dataset.block_split_info['block_3']['centroid'][1])
        c4 = np.array(dataset.block_split_info['block_4']['centroid'][1])
        
        np.linalg.norm(c0-c1, axis=-1)


    unit_test()