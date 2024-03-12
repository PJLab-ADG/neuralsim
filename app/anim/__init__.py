from app.resources import Scene

from .anim import Anim
from .file_anim import FileAnim
from .reverse_ego_anim import ReverseEgoAnim


def create_anim(anim_file_or_type: str, scene: Scene) -> Anim:
    if anim_file_or_type == "reverse_ego":
        return ReverseEgoAnim(scene)
    else:
        return FileAnim(anim_file_or_type, scene)
