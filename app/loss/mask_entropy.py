"""
@file   mask_entropy.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Close-range vs. distant-view volume-rendererd mask entropy regularization.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.render.pack_ops import packed_sum
from nr3d_lib.models.annealers import get_annealer

from app.resources import Scene

class MaskEntropyRegLoss(nn.Module):
    def __init__(
        self, 
        w: float, anneal: ConfigDict = None, 
        mode: str='crisp_cr', eps: float = 1.0e-5, 
        drawable_class_names: List[str] = [], 
        enable_after: int = 0) -> None:
        """ 
        Entropy regularization applied to the predicted opacity (i.e., mask) 
            to help disentangle close-range and distant-view models. 

        Args:
            w (float): Loss weight.
            anneal (ConfigDict, optional): Configuration for weight annealing. Defaults to None.
            mode (str, optional): The mode of entropy loss. Defaults to 'crisp_cr'.
            eps (float, optional): Numerical safety epsilon. Defaults to 1.0e-5.
            drawable_class_names (List[str], optional): All trainable model class names (currently not used). Defaults to [].
            enable_after (int, optional): Enable this loss after this iteration. Defaults to 0.
        """
        
        super().__init__()
        self.w = w
        self.eps = eps
        self.w_fn = None if anneal is None else get_annealer(**anneal)
        self.mode = mode
        self.enable_after = enable_after

    def forward_code_single(self, scene: Scene, ret: dict, rays_prefix, it: int) -> Dict[str, torch.Tensor]:
        if (self.mode == 'nop') or (it < self.enable_after):
            return dict()
        
        w = self.w if self.w_fn is None else self.w_fn(it=it)
        mask_pred = ret['rendered']['mask_volume']
        eps = self.eps
        
        if self.mode not in ['crisp']:
            raw_per_obj = ret['raw_per_obj']
            assert 'Distant' in scene.drawable_groups_by_class_name.keys(), f"mask_entropy mode={self.mode} only functions in cr-dv joint training."
            main_class_name = scene.main_class_name
            dv_class_name = 'Distant'
            cr_obj_id = scene.drawable_groups_by_class_name[main_class_name][0].id
            dv_obj_id = scene.drawable_groups_by_class_name[dv_class_name][0].id

            def volume_render_mask(buffer: dict):
                mask = torch.zeros(rays_prefix, device=scene.device, dtype=torch.float)
                if buffer['buffer_type'] != 'empty':
                    mask.index_put_((buffer['ray_inds_hit'],), packed_sum(
                        buffer['vw_in_total'], # NOTE: Should use `vw_in_total` instead of `vw`
                        buffer['pack_infos_collect']))
                return mask

            if 'cr' in self.mode:
                # Sum of visibility weights of cr in total rendering
                mask_pred_cr = volume_render_mask(raw_per_obj[cr_obj_id]['volume_buffer'])
            
            if 'dv' in self.mode:
                # Sum of visibility weights of dv in total rendering
                mask_pred_dv = volume_render_mask(raw_per_obj[dv_obj_id]['volume_buffer'])
        
        if self.mode == 'cross_cr_on_dv':
            loss = (mask_pred_cr * torch.log(mask_pred_dv.clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.cross_cr_on_dv': w * loss}
        elif self.mode == 'cross_cr_detached_on_dv':
            loss = (mask_pred_cr.detach() * torch.log(mask_pred_dv.clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.cross_cr_on_dv': w * loss}
        
        elif self.mode == 'cross_dv_on_cr':
            loss = (mask_pred_dv * torch.log(mask_pred_cr.clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.cross_dv_on_cr': w * loss}
        elif self.mode == 'cross_dv_detached_on_cr':
            loss = (mask_pred_dv.detach() * torch.log(mask_pred_cr.clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.cross_dv_on_cr': w * loss}

        elif self.mode == 'cross_crdv':
            loss1 = (mask_pred_cr * torch.log(mask_pred_dv.clamp_min(eps))).mean()
            loss2 = (mask_pred_dv * torch.log(mask_pred_cr.clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.cross_cr_on_dv': w * loss1, 'loss_mask_entropy.cross_dv_on_cr': w * loss2}
        elif self.mode == 'cross_crdv_detached':
            loss1 = (mask_pred_cr.detach() * torch.log(mask_pred_dv.clamp_min(eps))).mean()
            loss2 = (mask_pred_dv.detach() * torch.log(mask_pred_cr.clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.cross_cr_on_dv': w * loss1, 'loss_mask_entropy.cross_dv_on_cr': w * loss2}
        
        elif self.mode == 'cross_cr_on_dv_detached_full':
            # NOTE: .data to detach mask_pred_dv
            loss = (mask_pred_cr * torch.log(mask_pred_dv.data.clamp_min(eps)) + (1-mask_pred_cr) * torch.log((1-mask_pred_dv.data).clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.cross_cr_on_cr': w * loss}
        elif self.mode == 'cross_dv_on_cr_detached_full':
            # NOTE: .data to detach mask_pred_cr
            loss = (mask_pred_dv * torch.log(mask_pred_cr.data.clamp_min(eps)) + (1-mask_pred_dv) * torch.log((1-mask_pred_cr.data).clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.cross_cr_on_dv': w * loss}
        elif self.mode == 'cross_crdv_detached_full':
            # NOTE: .data to detach mask_pred_cr / mask_pred_dv respectively
            loss1 = (mask_pred_cr * torch.log(mask_pred_dv.data.clamp_min(eps)) + (1-mask_pred_cr) * torch.log((1-mask_pred_dv.data).clamp_min(eps))).mean()
            loss2 = (mask_pred_dv * torch.log(mask_pred_cr.data.clamp_min(eps)) + (1-mask_pred_dv) * torch.log((1-mask_pred_cr.data).clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.cross_crdv.1': w * loss1, 'loss_mask_entropy.cross_crdv.2': w * loss2}
        
        elif self.mode == 'crisp_cr':
            # loss = (-mask_pred_cr * torch.log(mask_pred_cr+eps)).mean()
            loss = -(mask_pred_cr * torch.log(mask_pred_cr.clamp_min(eps)) + (1-mask_pred_cr) * torch.log((1-mask_pred_cr).clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.crisp_cr': w * loss}
        elif self.mode == 'crisp_cr_0':
            # Entropy, but more preferably encourages to be 0.
            loss = -((1-mask_pred_cr) * torch.log((1-mask_pred_cr).clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.crisp_cr': w * loss}
        elif self.mode == 'crisp_cr_1':
            # Entropy, but more preferably encourages to be 1.
            loss = -(mask_pred_cr * torch.log(mask_pred_cr.clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.crisp_cr': w * loss}
        
        elif self.mode == 'crisp_dv':
            # loss = -(mask_pred_dv * torch.log(mask_pred_dv+eps)).mean()
            loss = -(mask_pred_dv * torch.log(mask_pred_dv.clamp_min(eps)) + (1-mask_pred_dv) * torch.log((1-mask_pred_dv).clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.crisp_dv': w * loss}
        elif self.mode == 'crisp_crdv':
            loss_cr = -(mask_pred_cr * torch.log(mask_pred_cr.clamp_min(eps)) + (1-mask_pred_cr) * torch.log((1-mask_pred_cr).clamp_min(eps))).mean()
            loss_dv = -(mask_pred_dv * torch.log(mask_pred_dv.clamp_min(eps)) + (1-mask_pred_dv) * torch.log((1-mask_pred_dv).clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.crisp_cr': w * loss_cr, 'loss_mask_entropy.crisp_dv': w * loss_dv}

        elif self.mode == 'crisp_cr_1_dv_0':
            loss_cr = -(mask_pred_cr * torch.log(mask_pred_cr.clamp_min(eps))).mean()
            loss_dv = -((1-mask_pred_dv) * torch.log((1-mask_pred_dv).clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.crisp_cr': w * loss_cr, 'loss_mask_entropy.crisp_dv': w * loss_dv}

        elif self.mode == 'crisp_cr_0_dv_1':
            loss_cr = -((1-mask_pred_cr) * torch.log((1-mask_pred_cr).clamp_min(eps))).mean()
            loss_dv = -(mask_pred_dv * torch.log(mask_pred_dv.clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy.crisp_cr': w * loss_cr, 'loss_mask_entropy.crisp_dv': w * loss_dv}

        elif self.mode == 'crisp_cr_and_cross_cr_detached_on_dv':
            # NOTE: not converged by now
            #       The main reason why it is not feasible is because the magnitude of the gradient in the intersecting part is much larger than that in the entropy part
            loss1 = -(mask_pred_cr * torch.log(mask_pred_cr.clamp_min(eps)) + (1-mask_pred_cr) * torch.log((1-mask_pred_cr).clamp_min(eps))).mean()
            loss2 = (mask_pred_cr.detach() * torch.log(mask_pred_dv.clamp_min(eps))).mean()
            ret_losses = {
                'loss_mask_entropy.crisp_cr': loss1, 
                'loss_mask_entropy.cross_cr_on_dv': loss2
            }

        elif self.mode == 'crisp':
            loss = -(mask_pred * torch.log(mask_pred.clamp_min(eps)) + (1-mask_pred) * torch.log((1-mask_pred).clamp_min(eps))).mean()
            ret_losses = {'loss_mask_entropy': w * loss}
        else:
            raise RuntimeError(f'Invalid mode={self.mode}')

        return ret_losses