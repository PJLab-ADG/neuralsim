from typing import Union
from .cameras import *
from .lidars import *
from .orth_camera import *
from .panaroma import *

OBSERVER_CLASS_NAMES = CAMERA_CLASS_NAMES + LIDAR_CLASS_NAMES
CAMERA_TYPE = Camera
LIDAR_TYPE = Union[Lidar, RaysLidar]
OBSERVER_TYPE = Union[Camera, Lidar, RaysLidar]