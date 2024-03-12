import json
import re
import torch
from typing import Dict

from nr3d_lib.models.attributes import *

from app.resources import Scene, SceneNode
from app.anim.anim import Anim


class FileAnim(Anim):
    n_frames: int
    time_step: float
    clip_range: range
    start_at_scene_frame: int
    pause_scene_anim: bool
    class_cfg: Dict[str, Any]
    _anim: Dict[str, Dict[str, torch.Tensor]]

    def __init__(self, file: str, scene: Scene):
        super().__init__(scene)
        with open(file, "r") as fp:
            json_data: dict = json.load(fp)
        
        self.time_step = json_data["time_step"]
        self.n_frames = len(json_data["frames"])
        self.clip_range = range(json_data.get("clip_start", 0),
                                json_data.get("clip_stop", self.n_frames))
        self.start_at_scene_frame = json_data.get("start_at_scene_frame", 0)
        self.pause_scene_anim = json_data.get("pause_scene_anim", 0)
        if self.pause_scene_anim is True:
            self.pause_scene_anim = self.clip_range[1] - self.clip_range[0]
        self.class_cfg = json_data["class_cfg"]
        
        self._anim = {}
        for i, frame in enumerate(json_data["frames"]):
            for key, value in frame.items():
                if isinstance(value, list):
                    if ":" not in key:
                        key = key + ":0"
                    id = self._find_or_create_node(scene, *key.split(":"))
                    self._add(i, id, value)
                else:
                    for short_id, trs in value.items():
                        id = self._find_or_create_node(scene, key, short_id)
                        self._add(i, id, trs)

    def slice_at(self, global_frame: int):
        anim_frame = global_frame - self.start_at_scene_frame
        scene_frame = global_frame - max(min(self.pause_scene_anim, anim_frame), 0)
        scene_frame = min(scene_frame, len(self.scene) - 1)
        self.scene.slice_at(scene_frame)
        
        for class_name, class_anim_cfg in self.class_cfg.items():
            if class_anim_cfg["merge_mode"] == "replace":
                for node in self.scene.all_nodes_by_class_name[class_name]:
                    node.i_valid = False
        
        if anim_frame >= 0 and anim_frame < self.clip_range.stop - self.clip_range.start:
            anim_frame_1 = anim_frame + self.clip_range.start
            for class_name, class_anim_cfg in self.class_cfg.items():
                for node in self.scene.all_nodes_by_class_name[class_name]:
                    if node.id in self._anim:
                        node.i_valid = True
                        node.i_valid_flags = self._anim[node.id]["valid_flag"][anim_frame_1]
                        node.transform = TransformMat4x4(self._anim[node.id]["transforms"][anim_frame_1])
                        if self._anim[node.id]["scales"][anim_frame_1, 0] > 0:
                            node.scale = Scale(self._anim[node.id]["scales"][anim_frame_1])
        
        self.scene.root.update()

    def _add(self, frame_ind: int, id: str, trs: List[float]):
        if id not in self._anim:
            self._anim[id] = {
                "valid_flag": torch.zeros(self.n_frames, dtype=torch.bool, device=self.device),
                "transforms": torch.zeros(self.n_frames, 4, 4, device=self.device),
                "scales": torch.zeros(self.n_frames, 3, device=self.device)
            }
        self._anim[id]["valid_flag"][frame_ind] = True
        self._anim[id]["transforms"][frame_ind] = torch.tensor(
            trs[:12] + [0., 0., 0., 1.], device=self.device).reshape(4, 4)
        if len(trs) == 15:
            self._anim[id]["scales"][frame_ind] = torch.tensor(trs[12:], device=self.device)

    def _find_or_create_node(self, scene: Scene, class_name: str, short_id: str):
        if re.match("^\d+$", short_id):
            return scene.all_nodes_by_class_name[class_name][int(short_id)].id
        for node in scene.all_nodes_by_class_name.get(class_name, []):
            if node.id.endswith(short_id):
                return node.id
        for scene_id, obj_id in scene.asset_bank.drawable_shared_map.get(class_name, []):
            if obj_id.endswith(short_id):
                new_node = SceneNode(obj_id, class_name, scene.device, scene.dtype)
                new_node_model_id = scene.asset_bank.asset_compute_id(obj=new_node, scene=scene, class_name=class_name)
                new_node.model = scene.asset_bank[new_node_model_id]
                scene.add_node(new_node)
                return obj_id   
        raise ValueError(f"{short_id} not exists in asset bank")
