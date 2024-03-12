import torch

from app.resources import Scene


class GridSpaceBuilder:

    def __call__(self, scene: Scene, far_clip: float, block_size: float) -> None:
        all_frustum_pts = scene.process_observer_infos(far_clip=far_clip).all_frustum_pts
        aabb = self.calculate_aabb(all_frustum_pts)
        grid_size = ((aabb[1] - aabb[0]) / block_size).ceil().long()
        grid_aabb = torch.stack([aabb[0], aabb[0] + grid_size * block_size])
        grids = torch.stack(torch.meshgrid([
            torch.arange(grid_size[0]),
            torch.arange(grid_size[1]),
            torch.arange(grid_size[2])], indexing="ij"), -1).to(scene.device)
        grid_corners = torch.stack(torch.meshgrid([
            torch.arange(grid_size[0] + 1),
            torch.arange(grid_size[1] + 1),
            torch.arange(grid_size[2] + 1)], indexing="ij"), -1).to(scene.device) * block_size + aabb[0]
        grid_flags = grid_size.new_zeros(*grid_size, dtype=torch.bool)
        scene.frozen_at_full_global_frame()
        for cam in scene.get_cameras(False):
            grid_corners_in_camera_space = cam.world_transform(
                grid_corners.expand(*cam.world_transform.prefix, -1, -1, -1, -1), True)  # (NC, NX + 1, NY + 1, NZ + 1, 3)
            grid_corners_in_image = cam.intr.proj(grid_corners_in_camera_space)
            grid_corners_in_image = torch.stack([
                grid_corners_in_image[0],
                grid_corners_in_image[1],
                grid_corners_in_camera_space[..., 2]
            ], -1)  # (NC, NX + 1, NY + 1, NZ + 1, 3)
            grid_edges_x_flags = self.aabb_ray_test(grid_corners_in_image[:, :-1],
                                                    grid_corners_in_image[:, 1:],
                                                    grid_corners_in_image.new_tensor([
                                                        [0., 0., 0.],
                                                        [cam.intr.W - 1, cam.intr.H - 1, far_clip]
                                                    ])).any(0) # (NX, NY + 1, NZ + 1)
            grid_edges_y_flags = self.aabb_ray_test(grid_corners_in_image[:, :, :-1],
                                                    grid_corners_in_image[:, :, 1:],
                                                    grid_corners_in_image.new_tensor([
                                                        [0., 0., 0.],
                                                        [cam.intr.W - 1, cam.intr.H - 1, far_clip]
                                                    ])).any(0) # (NX + 1, NY, NZ + 1)
            grid_edges_z_flags = self.aabb_ray_test(grid_corners_in_image[:, :, :, :-1],
                                                    grid_corners_in_image[:, :, :, 1:],
                                                    grid_corners_in_image.new_tensor([
                                                        [0., 0., 0.],
                                                        [cam.intr.W - 1, cam.intr.H - 1, far_clip]
                                                    ])).any(0) # (NX + 1, NY + 1, NZ)
            grid_flags.logical_or_(grid_edges_x_flags[:, :-1, :-1])
            grid_flags.logical_or_(grid_edges_x_flags[:, :-1, 1:])
            grid_flags.logical_or_(grid_edges_x_flags[:, 1:, :-1])
            grid_flags.logical_or_(grid_edges_x_flags[:, 1:, 1:])
            grid_flags.logical_or_(grid_edges_y_flags[:-1, :, :-1])
            grid_flags.logical_or_(grid_edges_y_flags[:-1, :, 1:])
            grid_flags.logical_or_(grid_edges_y_flags[1:, :, :-1])
            grid_flags.logical_or_(grid_edges_y_flags[1:, :, 1:])
            grid_flags.logical_or_(grid_edges_z_flags[:-1, :-1])
            grid_flags.logical_or_(grid_edges_z_flags[:-1, 1:])
            grid_flags.logical_or_(grid_edges_z_flags[1:, :-1])
            grid_flags.logical_or_(grid_edges_z_flags[1:, 1:])
            
        scene.unfrozen()
        return aabb, grid_aabb, grid_size, grid_flags

    def calculate_aabb(self, pts: torch.Tensor):
        """
        TODO: 
        The current implementation directly copies the bounding box calculation from neus.py, which needs further optimization.
        Original comment by Jianfei:
        A more scientific approach here would be to find the smallest enclosing rectangular box (axis not necessarily aligned BB) of these frustums.
        Currently, a simpler method is used, which may waste space on certain sequences.
        """
        bmin = pts.view(-1, 3).min(0).values
        bmax = pts.view(-1, 3).max(0).values
        return torch.stack([bmin, bmax], 0)

    def aabb_contains(self, pts: torch.Tensor, aabb: torch.Tensor):
        return torch.logical_and(pts >= aabb[0], pts <= aabb[1]).all(-1)

    def aabb_ray_test(self, x0: torch.Tensor, x1: torch.Tensor, aabb: torch.Tensor):
        """
        _summary_

        :param x0 `Tensor(N..., 3)`: _description_
        :param x1 `Tensor(N..., 3)`: _description_
        :param aabb `Tensor(2, 3)`: _description_
        :return `Tensor(N...)`
        """
        t0 = (aabb[0] - x0) / (x1 - x0)
        t1 = (aabb[1] - x0) / (x1 - x0)
        t_near = torch.minimum(t0, t1).max(-1).values
        t_far = torch.maximum(t0, t1).min(-1).values
        t_near = t_near.clamp_min(0.)
        t_far = t_far.clamp_max(1.)
        return t_far > t_near


if __name__ == "__main__":

    def test_space_builder():
        from dataio.mega_nerf import MegaNeRFDataset
        from nr3d_lib.plot import vis_occgrid_voxels_o3d, vis_camera_o3d
        import matplotlib.pyplot as plt

        extent_size = 0.5
        scene = Scene(device=torch.device('cuda'))
        scene.load_from_scenario(
            "/home/dengnianchen/Work/neuralsim/logs/mega/mill19/building/scenarios/main.pt", device=torch.device('cuda'))

        aabb, grid_aabb, grid_size, occ_grid = GridSpaceBuilder()(scene, extent_size, 0.5)

        scene.frozen_at_full_global_frame()
        cams = scene.get_cameras(False)
        #fig = plt.figure()
        #ax = plt.axes(projection="3d")
        cameras_vis = vis_camera_o3d([{
            "img_wh": (cams[0].intr.W, cams[0].intr.H),
            "intr": cams[0].intr.mat_3x3().cpu().numpy()[i],
            "c2w": cams[0].world_transform.mat_4x4().cpu().numpy()[i]
        } for i in range(len(scene))], cam_size=extent_size, show=False)
        scene.unfrozen()

        import open3d as o3d
        import open3d.visualization.gui as gui
        app = gui.Application.instance
        app.initialize()
        w = app.create_window(f"spc visualization", 1024, 768)
        widget3d = gui.SceneWidget()
        w.add_child(widget3d)
        widget3d.scene = o3d.visualization.rendering.Open3DScene(w.renderer)

        geos = []
        geo_occgrid = vis_occgrid_voxels_o3d(
            occ_grid, aabb=grid_aabb, show=False, draw_mesh=False)[0]
        geo_aabb = o3d.geometry.AxisAlignedBoundingBox(aabb[0].cpu().numpy(), aabb[1].cpu().numpy())
        geo_aabb.color = (0, 0, 0)
        geo_grid_aabb = o3d.geometry.AxisAlignedBoundingBox(
            grid_aabb[0].cpu().numpy(), grid_aabb[1].cpu().numpy())
        geo_grid_aabb.color = (0, 0, 0.5)
        geos += [geo_occgrid, geo_aabb, geo_grid_aabb]

        #for gi in range(grid_corners.shape[0]):
        #    geo_gridpoint = o3d.geometry.TriangleMesh.create_sphere(0.05, 10)
        #    geo_gridpoint.paint_uniform_color((1, 0, 0))
        #    geo_gridpoint.translate(grid_corners[gi].cpu().numpy())
        #    geos.append(geo_gridpoint)
        #widget3d.scene.add_geometry("occ_grid", lineset, line_mat)
        # for i, geo in enumerate(cameras_vis):
        #    widget3d.scene.add_geometry(f"camvis_{i}", geo)
        o3d.visualization.draw_geometries(cameras_vis + geos)
        app.run()
        # vis_camera_mplot(ax, cams[0].intr.mat_3x3().cpu().numpy(),
        #                 cams[0].world_transform.mat_4x4().cpu().numpy(),
        #                 cams[0].intr.H, cams[0].intr.W, annotation=False)
        # plt.show()

    test_space_builder()
