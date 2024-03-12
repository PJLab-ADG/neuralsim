import os
import torch
import plyfile
import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from collections import defaultdict

from nr3d_lib.plot import create_camera_frustum_o3d
from nr3d_lib.config import BaseConfig
from nr3d_lib.checkpoint import sorted_ckpts
from nr3d_lib.fmt import log
from nr3d_lib.utils import IDListedDict

from app.resources import AssetBank, SceneNode, Scene, load_scene_bank
from app.resources.observers import OBSERVER_CLASS_NAMES, Camera, Lidar, RaysLidar
from app.anim import create_anim
from app.resources import load_scenes_and_assets


##########################

if __name__ == "__main__":

    args = None

    def load_mesh(node:SceneNode):
        filenames = glob.glob(os.path.join(args.exp_dir, f"meshes/{node.class_name}*.ply"))
        for filename in filenames:
            if len(filenames) == 1 or filename[:-4].endswith(node.id):
                return o3d.io.read_triangle_mesh(filename)
        return None

    def main():
        #---------------------------------------------
        #--------------     Load     -----------------
        #---------------------------------------------
        device = torch.device('cuda')
        scene_bank, *_ = load_scenes_and_assets(**args, device=device)
        scene = scene_bank[0]

        anim = create_anim(args.anim, scene) if args.anim else None
        
        app = gui.Application.instance
        app.initialize()
        w = app.create_window(f"Scene Visualizer: {scene.id}", 1024, 768)
        em = w.theme.font_size
        
        # Create sidebar widget
        lb_label_mode = gui.Label("Label:")
        cb_label_mode = gui.Combobox()
        cb_label_mode.add_item("Hide")
        cb_label_mode.add_item("Class only")
        cb_label_mode.add_item("Full")
        cb_label_mode.selected_text = "Full"

        lb_view = gui.Label("View:")
        cb_view = gui.Combobox()
        cb_view.add_item("Switch to top-down")
        cb_view.add_item("Track main camera")
        cb_view.selected_text = "Switch to top-down"
        def cb_view_on_selection_changed_callback(item, index):
            if item == "Switch to top-down":
                widget3d.setup_camera(60.0, scene_aabb, scene_aabb.get_center())
        cb_view.set_on_selection_changed(cb_view_on_selection_changed_callback)

        checkbox_show_mesh = gui.Checkbox("Show mesh")

        lb_frame_inds = gui.Label("Frame:")
        slider_frame_inds = gui.Slider(gui.Slider.Type.INT)
        slider_frame_inds.set_limits(
            0, max(len(scene), (len(anim.clip_range) + anim.start_at_scene_frame) if anim else 0) - 1)
        
        checkbox_auto_play = gui.Checkbox("Auto play")

        btn_replay = gui.Button("Replay")
        def btn_replay_on_click_callback():
            slider_frame_inds.int_value = 0
        btn_replay.set_on_clicked(btn_replay_on_click_callback)

        sidebar = gui.Vert(.5 * em, gui.Margins(em, em, em, em))
        sidebar.add_child(lb_label_mode)
        sidebar.add_child(cb_label_mode)
        sidebar.add_child(lb_view)
        sidebar.add_child(cb_view)
        sidebar.add_child(checkbox_show_mesh)
        sidebar.add_child(lb_frame_inds)
        sidebar.add_child(slider_frame_inds)
        sidebar.add_child(checkbox_auto_play)
        sidebar.add_child(btn_replay)
        
        # Create scene widget
        widget3d = gui.SceneWidget()
        widget3d.scene = o3d.visualization.rendering.Open3DScene(w.renderer)

        # Add sidbar and scene widget to window
        w.add_child(widget3d)
        w.add_child(sidebar)

        # Register layout callback
        def on_layout_callback(layout_context):
            win_rect = w.content_rect
            widget3d.frame = win_rect
            sidebar_preferred_size = sidebar.calc_preferred_size(layout_context,
                                                                 gui.Widget.Constraints())
            panel_width = max(sidebar_preferred_size.width, 15 * em)
            panel_height = win_rect.height
            sidebar.frame = gui.Rect(win_rect.x, win_rect.y, panel_width, panel_height)
        
        w.set_on_layout(on_layout_callback)

        font_scale = 0.75
        arrow_length = 4.0
        # red = o3d.visualization.rendering.MaterialRecord()
        # red.base_color = [1., 0., 0., 1.0]
        # red.shader = "defaultUnlit"
        # green = o3d.visualization.rendering.MaterialRecord()
        # green.base_color = [0., 1., 0., 1.0]
        # green.shader = "defaultUnlit"
        # blue = o3d.visualization.rendering.MaterialRecord()
        # blue.base_color = [0., 0., 1., 1.0]
        # blue.shader = "defaultUnlit"
        # black = o3d.visualization.rendering.MaterialRecord()
        # black.base_color = [0., 0., 0., 1.]
        # black.line_width = 10
        # black.shader = "defaultUnlit"

        line_mat = o3d.visualization.rendering.MaterialRecord()
        line_mat.line_width = 1.5
        line_mat.shader = "unlitLine"

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        
        # if frame_ind is not None:
        #     scene.slice_at(frame_ind)
        cam0: Camera = scene.get_observer_groups_by_class_name('Camera', False)['camera_FRONT']
        cam0.far = 120.0
        all_drawables = scene.get_drawables(True)
        #filtered_drawables = IDListedDict(cam0.filter_drawable_groups(all_drawables))
        filtered_drawables = IDListedDict()

        slider_frame_inds.int_value = 0
        (anim or scene).slice_at(slider_frame_inds.int_value)
        geos = defaultdict(dict)
        meshes = {}
        labels = {}
        conn_geos_to_parent = {}

        def add_geometry(node: SceneNode, name: str, geometry, material):
            widget3d.scene.add_geometry(name, geometry, material)
            if node.i_valid:
                widget3d.scene.set_geometry_transform(
                    name, node.world_transform.mat_4x4().cpu().numpy())
            else:
                widget3d.scene.show_geometry(name, False)
            geos[node.id][name] = geometry

        for node in scene.all_nodes:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=arrow_length)
            add_geometry(node, node.id, coord_frame, mat)

            if checkbox_show_mesh.checked:
                if mesh := load_mesh(node):
                    add_geometry(node, f"{node.id}.mesh", mesh, mat)
                meshes[node.id] = mesh
            
            label_mode = cb_label_mode.selected_text
            if label_mode != "Hide" and node.i_valid:
                label_text = node.class_name if label_mode == "Class only" else \
                    node.class_name + ' : ' + node.id
                if node.class_name in OBSERVER_CLASS_NAMES:
                    l = widget3d.add_3d_label(
                        node.world_transform(torch.tensor(
                            [0.,0.,arrow_length], device=scene.device, dtype=scene.dtype
                        )).cpu().numpy(), 
                        label_text)
                else:
                    l = widget3d.add_3d_label(node.world_transform.translation().cpu().numpy(),
                                              label_text)
                l.scale = font_scale
                labels[node.id] = l

            # Draw Camera frustum
            if node.id == cam0.id:
                geometry_camera = create_camera_frustum_o3d(
                    img_wh=(cam0.intr.W, cam0.intr.H), 
                    intr=cam0.intr.mat_4x4().data.cpu().numpy(), 
                    c2w=np.eye(4), frustum_length=cam0.far, color=[0,0,1])
                add_geometry(node, f"{cam0.id}.frustum", geometry_camera, line_mat)
                
            # Draw OBB, if needed
            if node.model is not None and getattr(node.model, 'space', None):
                OBB = o3d.geometry.AxisAlignedBoundingBox(
                    node.scale(node.model.space.aabb[0]).cpu().numpy(),
                    node.scale(node.model.space.aabb[1]).cpu().numpy(),
                )
                if node.id in filtered_drawables.keys():
                    OBB.color = [1.0,0.0,0.0] # Drawables remained after frustum culling
                else:
                    OBB.color = [0.0,0.0,0.0] # Drawables culled away
                add_geometry(node, f"{node.id}.box", OBB, line_mat)
            # if node.model_bounding_sphere is not None:
            #     OBB = o3d.geometry.AxisAlignedBoundingBox(
            #         -node.scale.value().cpu().numpy() / 2.,
            #         node.scale.value().cpu().numpy() / 2.,
            #     )
            #     if node.id in filtered_drawables.keys():
            #         OBB.color = [1.0,0.0,0.0] # Drawables remained after frustum culling
            #     else:
            #         OBB.color = [0.0,0.0,0.0] # Drawables culled away
            #     add_geometry(node, f"{node.id}.box", OBB, line_mat)

            # Draw parent-child connection, if needed
            # if node.parent is not None:
            #     p_center = node.parent.world_transform.translation().data.cpu().numpy()
            #     o_center = node.world_transform.translation().data.cpu().numpy()
            #     if np.linalg.norm(p_center - o_center) > 0.01:
            #         points = [p_center, o_center]
            #         lines = [[0,1]]
            #         colors = [[0.7,0.7,0.7]]
            #         connection = o3d.geometry.LineSet()
            #         connection.points = o3d.utility.Vector3dVector(points)
            #         connection.lines = o3d.utility.Vector2iVector(lines)
            #         connection.colors = o3d.utility.Vector3dVector(colors)
            #         widget3d.scene.add_geometry(f"{node.parent.id}-{node.id}", connection, line_mat)
            #         conn_geos_to_parent[node.id] = connection
        scene.unfrozen()
        
        
        scene_aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(
                scene.process_observer_infos().all_frustum_pts.reshape(-1, 3).cpu().numpy()
            )
        )
        scene_aabb.color = [1.0, 0.5, 0.0]
        widget3d.scene.add_geometry("scene aabb", scene_aabb, line_mat)
        #widget3d.look_at(scene_aabb.get_center(),
        #                 [scene_aabb.get_center()[0], *scene_aabb.max_bound[1:]],
        #                 [0, 0, 1])
        widget3d.setup_camera(60.0, scene_aabb, scene_aabb.get_center())

        frame_ind = 1
        def on_tick_event_callback():
            nonlocal frame_ind
            frame_ind = min(frame_ind, int(slider_frame_inds.get_maximum_value))
            (anim or scene).slice_at(slider_frame_inds.int_value)
            if cb_view.selected_text == "Track main camera":
                cam_trs = cam0.world_transform.tensor
                widget3d.look_at(cam_trs[:3, 3].cpu().numpy(),
                                 (cam_trs[:3, 3] + cam_trs[:3, 2]).cpu().numpy(),
                                 (-cam_trs[:3, 1]).cpu().numpy())
            for node in scene.all_nodes:
                if checkbox_show_mesh.checked:
                    if node.id not in meshes:
                        if mesh := load_mesh(node):
                            add_geometry(node, f"{node.id}.mesh", mesh, mat)
                        meshes[node.id] = mesh
                for geo_name in geos[node.id]:
                    widget3d.scene.show_geometry(geo_name, node.i_valid)
                    widget3d.scene.set_geometry_transform(
                        geo_name, node.world_transform.mat_4x4().cpu().numpy())
                if not checkbox_show_mesh.checked:
                    widget3d.scene.show_geometry(f"{node.id}.mesh", False)
                if node.id == cam0.id and cb_view.selected_text == "Track main camera":
                    for geo_name in geos[node.id]:
                        widget3d.scene.show_geometry(geo_name, False)
                
                label_mode = cb_label_mode.selected_text
                if not node.i_valid or label_mode == "Hide":
                    if node.id in labels:
                        widget3d.remove_3d_label(labels[node.id])
                        del labels[node.id]
                else:
                    label_text = node.class_name if label_mode == "Class only" else \
                        node.class_name + ' : ' + node.id
                    if node.class_name in OBSERVER_CLASS_NAMES:
                        label_position = node.world_transform(
                            torch.tensor([0.,0.,arrow_length], device=scene.device, dtype=scene.dtype)
                        ).cpu().numpy()
                    else:
                        label_position = node.world_transform.translation().cpu().numpy()
                    if node.id in labels:
                        labels[node.id].position = label_position
                        labels[node.id].text = label_text
                    else:
                        labels[node.id] = widget3d.add_3d_label(label_position, label_text)
                        labels[node.id].scale = font_scale
            scene.unfrozen()
            if checkbox_auto_play.checked:
                slider_frame_inds.int_value += 1
            return True
        w.set_on_tick_event(on_tick_event_callback)
        app.run()


    bc = BaseConfig()
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument('--start_frame', type=int, default=25)
    bc.parser.add_argument("--num_frames", type=int, default=None)
    bc.parser.add_argument('--stop_frame', type=int, default=125)
    bc.parser.add_argument('--anim', type=str)
    args = bc.parse()
    main()