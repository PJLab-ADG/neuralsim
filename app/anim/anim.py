import torch
from app.resources import Scene


class Anim:
    scene: Scene
    device: torch.device

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.device = scene.device

    def slice_at(self, global_frame: int):
        raise NotImplementedError()
