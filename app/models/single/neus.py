"""
@file   neus.py
@author Jianfei Guo, Shanghai AI Lab
@brief  neuralsim's API for NeuS models.
"""

__all__ = [
    'LoTDNeuSObj', 
    'MLPNeuSObj',
    'LoTDNeuSStreet', 
    'MLPNeuSStreet'
]


import torch

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.fields.neus import MlpPENeuSModel, LoTDNeuSModel
from nr3d_lib.models.fields.sdf import pretrain_sdf_capsule, pretrain_sdf_road_surface

from app.models.base import AssetAssignment, AssetMixin
from app.resources import Scene, SceneNode
from app.resources.observers import Camera

class LoTDNeuSObj(AssetMixin, LoTDNeuSModel):
    """
    NeuS network for single object-centric scene or indoor scene, represented by LoTD encodings
    
    MRO: LoTDNeuSObj -> AssetMixin -> LoTDNeuSModel -> NeusRendererMixin -> LoTDNeuS -> ModelMixin -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    @torch.no_grad()
    def populate(self, scene: Scene = None, obj: SceneNode = None, config: ConfigDict = None, 
                 dtype=torch.float, device=torch.device('cuda'), **kwargs):
        LoTDNeuSModel.populate(self)
    
    def initialize(self, scene:Scene, obj:SceneNode, config: ConfigDict, logger: Logger=None, log_prefix: str=None):
        # self.grad_guard_when_render.logger = logger
        # self.grad_guard_when_uniform.logger = logger
        return self.implicit_surface.initialize(config=config, logger=logger, log_prefix=log_prefix)

    @classmethod
    def compute_model_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

class MLPNeuSObj(AssetMixin, MlpPENeuSModel):
    """
    NeuS network for single object-centric or indoor scene, represented by MLP
    
    MRO: MLPNeuSObj -> AssetMixin -> MlpPENeuSModel -> NeusRendererMixin -> MlpPENeuS -> ModelMixin -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    @torch.no_grad()
    def populate(self, scene: Scene = None, obj: SceneNode = None, config: ConfigDict = None, 
                 dtype=torch.float, device=torch.device('cuda'), **kwargs):
        MlpPENeuSModel.populate(self)
    
    def initialize(self, scene:Scene, obj:SceneNode, config: ConfigDict, logger: Logger=None, log_prefix: str=None):
        return self.implicit_surface.initialize(config=config, logger=logger, log_prefix=log_prefix)

    @classmethod
    def compute_model_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

class LoTDNeuSStreet(AssetMixin, LoTDNeuSModel):
    """
    NeuS network for single street-view scene, reprensented by LoTD encodings
    
    MRO: LoTDNeuSStreet -> AssetMixin -> LoTDNeuSModel -> NeusRendererMixin -> LoTDNeuS -> ModelMixin -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    @torch.no_grad()
    def populate(self, scene: Scene = None, obj: SceneNode = None, config: ConfigDict = None, 
                 dtype=torch.float, device=torch.device('cuda'), **kwargs):
        """
        1. Use the range of observers in the scene to determine the pose and scale of the obj, so that the network input is automatically unit input
            For more descriptions, please refer to StreetSurf paper section 3.1
        2. If there is a need to change or additionally assign some attributes to the obj
        """
        
        extend_size = config.get('extend_size')
        use_cuboid = config.get('use_cuboid', True)
        
        frustum_extend_pts = []
        cams = scene.get_cameras(only_valid=False)
        scene.frozen_at_full()
        for cam in cams:
            frustum = cam.get_view_frustum_pts(near=0., far=extend_size)
            frustum_extend_pts.append(frustum)
        frustum_extend_pts = torch.stack(frustum_extend_pts, 0)
        scene.unfrozen()
        
        xyz_extend = frustum_extend_pts.view(-1,3)
        
        # NOTE: Apply pre-loaded transform from dataset
        scene.frozen_at(0)
        xyz_extend = obj.world_transform.forward(xyz_extend, inv=True) / obj.scale.ratio() # From world to street_obj
        scene.unfrozen()
        
        """
        The more scientific approach here is to find the smallest enclosing rectangular prism of these view cones (the axes of the BB do not necessarily have to be aligned).
        Currently, a rather crude method is used, which may waste a some space on certain sequences.
        """
        bmin = xyz_extend.min(0).values
        bmax = xyz_extend.max(0).values
        
        if use_cuboid:
            print(f"=> {obj.id} using cuboid space")
            aabb = torch.stack([bmin, bmax], 0)
        else:
            print(f"=> {obj.id} using cubic space")
            radius = (bmax - bmin).max().item()
            center = (bmax + bmin) / 2.
            aabb = torch.stack([center - radius, center + radius], 0)
        
        LoTDNeuSModel.populate(self, aabb=aabb)

    def initialize(self, scene: Scene, obj: SceneNode, config: ConfigDict, logger=None, log_prefix=None):
        if config is None: config = ConfigDict()
        geo_init_method = self.implicit_surface.geo_init_method
        if geo_init_method == 'pretrain' or ('pretrain_after' in geo_init_method):
            if not self.implicit_surface.is_pretrained:
                if log_prefix is None:
                    log_prefix = obj.id
                
                config = config.copy()
                obs_ref = config.pop('obs_ref')
                target_shape = config.pop('target_shape', 'capsule')
                
                with torch.no_grad():
                    scene.frozen_at_full()
                    obs = scene.observers[obs_ref]
                    tracks = obs.world_transform.translation()
                    scene.unfrozen()
                    
                    scene.frozen_at(0)
                    tracks_in_obj = obj.world_transform(tracks, inv=True) / obj.scale.ratio()
                    scene.unfrozen()

                if target_shape == 'capsule':
                    pretrain_sdf_capsule(self.implicit_surface, tracks_in_obj, **config, logger=logger, log_prefix=log_prefix)
                elif target_shape == 'road_surface':
                    pretrain_sdf_road_surface(self.implicit_surface, tracks_in_obj, **config, logger=logger, log_prefix=log_prefix)
                else:
                    raise RuntimeError(f'Invalid target_shape={target_shape}')

                self.implicit_surface.is_pretrained = ~self.implicit_surface.is_pretrained
                return True
        return False

    def preprocess_per_render_frame(self, renderer, observer: Camera, per_frame_info: dict={}):
        pass

    @torch.no_grad()
    def val(self, scene: Scene = None, obj: SceneNode = None, it: int = ..., logger: Logger = None, log_prefix: str = ''):
        pass
        # mesh = self.accel.debug_get_mesh()
        # logger.add_open3d(scene.id, ".".join([obj.model.id, "accel"]), mesh, it)
        
        # # verts, faces = self.accel.debug_get_mesh()
        # mesh = geometries[0]
        # verts = torch.tensor(mesh.vertices, dtype=torch.float, device=self.device)
        # faces = torch.tensor(mesh.triangles, dtype=torch.float, device=self.device)
        # logger.add_mesh(scene.id, ".".join([obj.model.id, "accel"]), verts, faces=faces, it=it)

    @classmethod
    def compute_model_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

class MLPNeuSStreet(AssetMixin, MlpPENeuSModel):
    """
    NeuS network for single street-view scene, reprensented by LoTD encodings
    
    MRO: MLPNeuSStreet -> AssetMixin -> MlpPENeuSModel -> NeusRendererMixin -> MlpPENeuS -> ModelMixin -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    @torch.no_grad()
    def populate(self, scene: Scene = None, obj: SceneNode = None, config: ConfigDict = None, 
                 dtype=torch.float, device=torch.device('cuda'), **kwargs):
        
        extend_size = config.get('extend_size')
        use_cuboid = config.get('use_cuboid', True)
        
        frustum_extend_pts = []
        cams = scene.get_cameras(only_valid=False)
        scene.frozen_at_full()
        for cam in cams:
            frustum = cam.get_view_frustum_pts(near=0., far=extend_size)
            frustum_extend_pts.append(frustum)
        frustum_extend_pts = torch.stack(frustum_extend_pts, 0)
        scene.unfrozen()
        
        xyz_extend = frustum_extend_pts.view(-1,3)
        
        # NOTE: Apply pre-loaded transform from dataset
        scene.frozen_at(0)
        xyz_extend = obj.world_transform.forward(xyz_extend, inv=True) / obj.scale.ratio() # From world to street_obj
        scene.unfrozen()

        bmin = xyz_extend.min(0).values
        bmax = xyz_extend.max(0).values

        if use_cuboid:
            print(f"=> {obj.id} using cuboid space")
            aabb = torch.stack([bmin, bmax], 0)
        else:
            print(f"=> {obj.id} using cubic space")
            radius = (bmax - bmin).max().item()
            center = (bmax + bmin) / 2.
            aabb = torch.stack([center - radius, center + radius], 0)

        MlpPENeuSModel.populate(self, aabb=aabb)

    def initialize(self, scene: Scene, obj: SceneNode, config: ConfigDict, logger=None, log_prefix=None):
        if config is None: config = ConfigDict()
        geo_init_method = self.implicit_surface.geo_init_method
        if geo_init_method == 'pretrain' or ('pretrain_after' in geo_init_method):
            if not self.implicit_surface.is_pretrained:
                if log_prefix is None:
                    log_prefix = obj.id
                
                config = config.copy()
                obs_ref = config.pop('obs_ref')
                target_shape = config.pop('target_shape', 'capsule')
                
                with torch.no_grad():
                    scene.frozen_at_full()
                    obs = scene.observers[obs_ref]
                    tracks = obs.world_transform.translation()
                    scene.unfrozen()
                    
                    scene.frozen_at(0)
                    tracks_in_obj = obj.world_transform(tracks, inv=True) / obj.scale.ratio()
                    scene.unfrozen()

                if target_shape == 'capsule':
                    pretrain_sdf_capsule(self.implicit_surface, tracks_in_obj, **config, logger=logger, log_prefix=log_prefix)
                elif target_shape == 'road_surface':
                    pretrain_sdf_road_surface(self.implicit_surface, tracks_in_obj, **config, logger=logger, log_prefix=log_prefix)
                else:
                    raise RuntimeError(f'Invalid target_shape={target_shape}')

                self.implicit_surface.is_pretrained = ~self.implicit_surface.is_pretrained
                return True
        return False

    @classmethod
    def compute_model_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"


if __name__ == "__main__":
    def unit_test():
        pass