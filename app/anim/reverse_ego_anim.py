import torch
from typing import Dict

from nr3d_lib.models.attributes import *

from app.resources import Scene
from app.anim.anim import Anim

class ReverseEgoAnim(Anim):
    n_frames: int
    time_step: float
    clip_range: range
    start_at_scene_frame: int
    pause_scene_anim: bool
    class_cfg: Dict[str, Any]
    _anim: Dict[str, Dict[str, torch.Tensor]]

    def __init__(self, scene: Scene):
        super().__init__(scene)

    def slice_at(self, global_frame: int):
        scene_frame = min(global_frame, len(self.scene) - 1)
        self.scene.slice_at(scene_frame)

        if "EgoVehicle" in self.scene.all_nodes_by_class_name:
            ego_node = self.scene.all_nodes_by_class_name["EgoVehicle"][0]
            ego_node._slice_at(len(self.scene) - 1 - scene_frame)
            ego_node.transform.tensor[..., :2] *= -1. # Flip x and y
            ego_node.update()
        else:
            raise NotImplementedError("Currently only support scene containing ego-vehicle node")