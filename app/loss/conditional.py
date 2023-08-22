
import numbers
import functools
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.loss.recon import l2_loss, relative_l2_loss, l1_loss
from nr3d_lib.models.loss.safe import safe_binary_cross_entropy
from nr3d_lib.models.annealers import get_annealer

from app.resources import Scene

class LatentLoss(nn.Module):
    pass

class DeformationLoss(nn.Module):
    pass

class ConditionalLosses(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self) -> Dict[str, torch.Tensor]:
        pass