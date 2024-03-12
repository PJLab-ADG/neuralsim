"""
@file   inspect_rendering.py
@author Nianchen Deng, Shanghai AI Lab
@brief  A tool implemented in Dash+Plotly for inspecting the rendering process of NR3D.
        To run this tool, please first install Dash and Plotly by running:

        ```bash
        pip install dash plotly pandas dash-ag-grid dash-bootstrap-components
        ```

        Then run this script by:

        ```bash
        python code_single/tools/inspect_rendering.py --resume_dir <path_to_exp_dir> [--downscale=4 --rayschunk=4096 --assetbank_cfg.Street.model_params.ray_query_cfg.query_mode=march_occ_multi_upsample]
        ```
"""
import os
import sys


def set_env(depth: int):
    # Add project root to sys.path
    current_file_path = os.path.abspath(__file__)
    project_root_path = os.path.dirname(current_file_path)
    for _ in range(depth):
        project_root_path = os.path.dirname(project_root_path)
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)
        print(f"Added {project_root_path} to sys.path")


set_env(2)

import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import dash_ag_grid as dag  # pip install dash-ag-grid
import torch.nn.functional as nn_f
from dash import dcc, html, callback, ctx
from dash.dependencies import Input, Output, State, ALL

from nr3d_lib.config import ConfigDict
from nr3d_lib.profile import Profiler, profile
from nr3d_lib.graphics.nerf import packed_alpha_to_vw, ray_alpha_to_vw
from nr3d_lib.models.spatial.aabb import AABBSpace
from nr3d_lib.models.accelerations import OccGridEma
from nr3d_lib.models.grid_encodings.lotd import lotd

from app.resources import load_scenes_and_assets
from app.resources.observers import Camera
from app.renderers import SingleVolumeRenderer


device = torch.device('cuda')
torch.set_grad_enabled(False)

scene = None
cam = None
obj = None
renderer = None
renderer_ret = None
rgb_fig = None
sp_rgb_fig = None
fg_query_cfg = None
x = None
y = None

sphere_trace_cfg = ConfigDict(distance_scale=30., min_step=.2, hit_threshold=1e-3,
                              max_march_iters=500, drop_alive_rate=0., tail_sample_threshold = 20000,
                              tail_sample_step_size = None)


def Header(name, app):
    title = html.H2(name, style={"margin-top": 5})
    return dbc.Row([dbc.Col(title, md=8)])


def unsnake(st):
    """BECAUSE_WE_DONT_READ_LIKE_THAT"""
    return st.replace("_", " ").title()


def img_to_fig(img: torch.Tensor):
    fig = px.imshow(img.cpu().numpy())
    fig.update_layout(
        margin=dict(l=10, r=10, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, range=(0, img.shape[1]))
    fig.update_yaxes(showticklabels=False, showgrid=False, range=(img.shape[0], 0))
    return fig


def render(cam, sphere_trace=False, return_channel="rgb"):
    global renderer_ret
    if sphere_trace:
        bypass_ray_query_cfg = ConfigDict({
            obj.class_name: ConfigDict({
                "query_mode": "sphere_trace",
                "query_param": ConfigDict(**sphere_trace_cfg, debug=True),
            })
        })
        renderer_ret = renderer.render(scene, observer=cam, render_per_obj_individual=True, rayschunk=0,
                                       bypass_ray_query_cfg=bypass_ray_query_cfg,
                                       with_normal=return_channel == "normal", only_cr=True)
    else:
        renderer_ret = renderer.render(scene, observer=cam, render_per_obj_individual=True,
                                       rayschunk=args.rayschunk,
                                       with_normal=return_channel == "normal", only_cr=True)
    if return_channel == "rgb":
        return renderer_ret["rendered"]["rgb_volume"].reshape(cam.intr.H, cam.intr.W, 3)
    elif return_channel == "normal":
        return (renderer_ret["rendered"]['normals_volume'] / 2 + 0.5).reshape(cam.intr.H, cam.intr.W, 3)


def query_sdf(pts: torch.Tensor):
    query_ret = obj_model.forward_sdf_nablas(pts)
    if isinstance(query_ret, dict):
        d = query_ret["sdf"]
        nablas = query_ret.get("nablas")
    else:
        d = query_ret
        nablas = None
    return d.reshape_as(pts[..., 0]), nablas.reshape_as(pts)


def color_sdf(d: torch.Tensor, include_iso=True):
    d = d.clip(-1.0, 1.0)
    blue = d.clip(0.0, 0.2) * 5.
    yellow = 1.0 - blue
    colors = d.new_zeros(*d.shape, 3)
    colors[:, 2] = blue
    colors += yellow[:, None] * d.new_tensor([0.4, 0.3, 0.0])
    colors += 0.2
    colors[d < 0] = d.new_tensor([1.0, 0.38, 0.0])
    if include_iso:
        for i in range(-50, 51):
            colors[(d - 0.02 * i).abs() < 0.0015] = 0.8
    colors[d.abs() < 0.001] = 0.0
    return colors


def plot_sdf(fig: go.Figure, ray_o: torch.Tensor, ray_d: torch.Tensor, rays_ts: torch.Tensor, near: float, far: float,
             n: int, plot_n: bool = False, plot_slope: bool = False):
    rays_o, rays_d = scene.convert_rays_in_node(ray_o[None], ray_d[None], obj)
    rays_o, rays_d = obj_space.normalize_rays(rays_o, rays_d)
    rays_tested = obj_space.ray_test(rays_o, rays_d, near, far, normalized=True)

    t = torch.linspace(rays_tested["near"][0], rays_tested["far"][0], n, device=ray_o.device)
    pts = ray_o + ray_d * t[:, None]
    pts_in_obj = obj.world_transform.forward(pts, inv=True)
    pts_in_obj = obj_space.normalize_coords(pts_in_obj)
    d, nablas = query_sdf(pts_in_obj)

    t = t.cpu().numpy()
    d = d.cpu().numpy()
    nablas = nablas.norm(dim=-1).cpu().numpy() if nablas is not None else None

    fig.add_trace(go.Scatter(x=t, y=d, mode='lines', name="SDF", showlegend=False))
    if plot_n:
        fig.add_trace(go.Scatter(x=t, y=nablas, mode='lines', name="SDF Grad", yaxis="y2",
                                 visible='legendonly'))
    if plot_slope:
        fig.add_trace(go.Scatter(x=t, y=np.abs(np.gradient(d, t)), mode='lines', name="SDF Slope",
                                 yaxis="y2", visible='legendonly'))
    fig.update_xaxes(title="Depth", type='linear',
                     range=[rays_tested["near"][0].item(), rays_tested["far"][0].item()])
    fig.update_yaxes(title="SDF", type='linear', range=[-0.03, 0.03])
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                      margin=dict(l=30, r=30, t=50, b=50),
                      yaxis2=dict(anchor="x", overlaying="y", side="right", range=[-0.05, 1.05]))


def plot_volume_render_samples(fig: go.Figure, camera: Camera, ray_o: torch.Tensor, ray_d: torch.Tensor, rays_ts: torch.Tensor):
    rays_o, rays_d = scene.convert_rays_in_node(ray_o[None], ray_d[None], obj)
    rays_o, rays_d = obj_space.normalize_rays(rays_o, rays_d)
    
    rays_extra_data = dict(rays_ts=rays_ts)
    if scene.image_embeddings is not None:
        # rays_ts = torch.tensor([camera.i], device=device)
        rays_extra_data['rays_h_appear'] = scene.image_embeddings[camera.id](rays_ts, mode='interp')
    rays_tested = obj_space.ray_test(rays_o, rays_d, camera.near, camera.far, normalized=True, **rays_extra_data)

    volume_buffer, details = obj.model._ray_query_march_occ_multi_upsample(
        rays_tested, with_normal=False, forward_inv_s=fg_query_cfg["forward_inv_s"],
        debug_query_data=(debug_query_data := {}), **fg_query_cfg["query_param"])
    if volume_buffer['type'] == "empty":
        return fig

    if "coarse" in debug_query_data:
        n = debug_query_data["coarse"]["depth"].numel()
        fig.add_trace(go.Scatter(x=debug_query_data["coarse"]["depth"].cpu().numpy(),
                                 y=debug_query_data["coarse"]["sdf"].cpu().numpy(),
                                 name=f"{n} Coarse Samples",
                                 mode='markers', marker=dict(color="silver")))
    if "ray_march" in debug_query_data:
        n = debug_query_data["ray_march"]["depth"].numel()
        fig.add_trace(go.Scatter(x=debug_query_data["ray_march"]["depth"].cpu().numpy(),
                                 y=debug_query_data["ray_march"]["sdf"].cpu().numpy(),
                                 name=f"{n} Ray-march Samples", mode='markers',
                                 marker=dict(color="gray")))
        fig.add_trace(go.Bar(x=debug_query_data["ray_march"]["depth"].cpu().numpy(),
                             y=debug_query_data["ray_march"]["pdf"].cpu().numpy(),
                             name="Ray-march PDF", yaxis="y2"))
    if "fine" in debug_query_data:
        sample_stage_colors = [
            "red" if s == 1 else "orange" if s == 2 else "yellow" if s == 3 else "green"
            for s in debug_query_data["fine"]["upsample_stages"].cpu().numpy()
        ]
        n = debug_query_data["fine"]["depth"].numel()
        fig.add_trace(go.Scatter(x=debug_query_data["fine"]["depth"].cpu().numpy(),
                                 y=debug_query_data["fine"]["sdf"].cpu().numpy(),
                                 name=f"{n} Up- Samples",
                                 mode='markers', marker=dict(color=sample_stage_colors)))

    sample_depths = volume_buffer["t"].flatten()
    sample_alphas = volume_buffer["opacity_alpha"].flatten()
    if volume_buffer['type'] == 'batched':
        sample_weights = ray_alpha_to_vw(sample_alphas)
    else:
        sample_weights = packed_alpha_to_vw(sample_alphas, volume_buffer["pack_infos_hit"])
    sample_cumweights = sample_weights.cumsum(dim=0)
    sample_weights_normed = sample_weights / (sample_weights.sum() + 1e-10)
    ray_depth = (sample_depths * sample_weights_normed).sum().item()

    sample_depths = sample_depths.cpu().numpy()
    sample_alphas = sample_alphas.cpu().numpy()
    sample_weights = sample_weights.cpu().numpy()
    sample_cumweights = sample_cumweights.cpu().numpy()
    fig.add_trace(go.Scatter(x=sample_depths, y=sample_alphas, mode='markers+lines',
                             name="Alpha", yaxis="y2"))
    # fig.add_trace(go.Scatter(x=sample_depths, y=sample_weights_normed, mode='markers+lines',
    #                          name="Normalized Weight", yaxis="y2"))
    fig.add_trace(go.Scatter(x=sample_depths, y=sample_cumweights, mode='lines',
                             name="Cum-Weight", yaxis="y2"))
    fig.update_layout(yaxis2=dict(anchor="x", overlaying="y", side="right", range=[-0.05, 1.05]))

    if sample_weights.sum() > 1e-3:
        fig.add_trace(go.Scatter(x=[ray_depth] * 2, y=[0., 1.],
                                 mode="lines", name="Depth-Predicted", yaxis="y2",
                                 line=go.scatter.Line(color="indianred", width=3)))


def plot_sphere_trace_samples(fig: go.Figure, camera: Camera, ray_o: torch.Tensor, ray_d: torch.Tensor, rays_ts: torch.Tensor):
    rays_o, rays_d = scene.convert_rays_in_node(ray_o[None], ray_d[None], obj)
    rays_o, rays_d = obj_space.normalize_rays(rays_o, rays_d)
    near = camera.near
    far = camera.far

    rays_extra_data = dict(rays_ts=rays_ts)
    if scene.image_embeddings is not None:
        # rays_ts = torch.tensor([camera.i], device=device)
        rays_extra_data['rays_h_appear'] = scene.image_embeddings[camera.id](rays_ts, mode='interp')
    ray_tested = obj_space.ray_test(rays_o, rays_d, near, far, normalized=True, **rays_extra_data)
    
    enable_profile_bk = lotd.global_configs["enable_profile"]
    lotd.configure(enable_profile=False)
    rays_hit = obj.model.tracer.trace(ray_tested, obj_model.forward_sdf, print_debug_log=True,
                                      debug_output=(debug_output := {}), debug_replay=True)
    lotd.configure(enable_profile=enable_profile_bk)
    
    for seg in debug_output["segs"]:
        fig.add_vrect(
            x0=seg[0].item(), x1=seg[1].item(),
            fillcolor="LightSalmon", opacity=0.5,
            layer="below", line_width=0,
        )

    trace_data = debug_output.get("trace_data", [])
    if (len(trace_data) > 0):
        trace_depths = np.array([
            trace_data[i]["rays_alive"]["t"][0].item()
            for i in range(len(trace_data))
        ])
        trace_sdfs = np.array([trace_data[i]["d"][0].item() for i in range(len(trace_data))])
        trace_n_steps = [
            trace_data[i]["rays_alive"]["n_steps"][0].item()
            for i in range(len(trace_data))
        ]
        trace_status = [
            ["ALIVE", "HIT", "OUT"][trace_data[i]["rays_alive"]["status"][0].item()]
            for i in range(len(trace_data))
        ]
        trace_hit_region_infos = np.array([
            trace_data[i]["rays_alive"]["hit_region_infos"][0].cpu().numpy()
            for i in range(len(trace_data))
        ])
        trace_seg_idxs = [
            trace_data[i]["rays_alive"]["seg_idxs"][0].item()
            for i in range(len(trace_data))
        ]
        trace_seg_end_idxs = [
            trace_data[i]["rays_alive"]["seg_end_idxs"][0].item()
            for i in range(len(trace_data))
        ]
        trace_debug_flags = [
            trace_data[i]["rays_alive"]["debug_flag"][0].item()
            for i in range(len(trace_data))
        ]
        if rays_hit["idx"].numel() > 0:
            rays_hit_d = obj_model.forward_sdf(rays_hit["pos"])
            trace_depths = np.concatenate([trace_depths, rays_hit["t"].cpu().numpy()])
            trace_sdfs = np.concatenate([trace_sdfs, rays_hit_d["sdf"].cpu().numpy()])
            trace_n_steps += [rays_hit["n_steps"][0].item()]
            trace_status += ["HIT*"]
            trace_debug_flags += [255]
            trace_hit_region_infos = np.concatenate([
                trace_hit_region_infos,
                np.zeros_like(trace_hit_region_infos[:1])
            ])
            trace_seg_idxs += [-1]
            trace_seg_end_idxs += [-1]

        for i in range(min(50, trace_depths.shape[0])):
            print(
                f"{trace_n_steps[i]}\t{trace_depths[i]:.4f}\t{trace_sdfs[i]:.5f}\t{trace_status[i]}\t"
                f"{trace_debug_flags[i]}\t{trace_hit_region_infos[i][0]:.4f}\t{trace_hit_region_infos[i][1]:.4f}\t"
                f"{trace_hit_region_infos[i][2]:.4f}\t{trace_hit_region_infos[i][3]:.4f}\t"
                f"{trace_seg_idxs[i]}\t{trace_seg_end_idxs[i]}")
        fig.add_trace(go.Scatter(x=trace_depths, y=trace_sdfs,
                                 name=f"{len(trace_depths)} Trace Samples", mode='markers',
                                 marker={"color": list(range(len(trace_depths)))}))

    fig.update_layout(yaxis2=dict(anchor="x", overlaying="y", side="right", range=[-0.05, 1.05]))
    if rays_hit["idx"].numel() > 0:
        fig.add_trace(go.Scatter(x=[rays_hit["t"][0].item()] * 2, y=[0., 1.],
                                 mode="lines", name="Depth-Predicted", yaxis="y2",
                                 line=go.scatter.Line(color="rebeccapurple", width=3)))


class SlicePlane:
    def __init__(self, rays_o: torch.Tensor, rays_d: torch.Tensor, length: float,
                 pix_width: int, pix_length: int, vertical: bool = False) -> None:
        self.length = length
        far_corner1 = rays_o[0] + rays_d[0] * length
        far_corner2 = rays_o[-1] + rays_d[-1] * length
        self.width = (far_corner2 - far_corner1).norm()
        rays_plane_x = nn_f.normalize(rays_d[-1] - rays_d[0], dim=0)
        rays_plane_z = nn_f.normalize(rays_d[-1] + rays_d[0], dim=0)
        rays_plane_y = torch.cross(rays_plane_z, rays_plane_x)
        self.rot = torch.stack([rays_plane_x, rays_plane_y, rays_plane_z], dim=-1)
        self.origin = rays_o[0]
        self.vertical = vertical
        self.resolution = [pix_length, pix_width] if vertical else [pix_width, pix_length]

        slice_img_x, slice_img_y = torch.meshgrid([
            torch.linspace(-1., 1., self.resolution[0], device=rays_o.device),
            torch.linspace(-1., 1., self.resolution[1], device=rays_o.device)
        ], indexing="xy")
        if self.vertical:
            img_local_pts = torch.stack([
                slice_img_y * self.width / 2.,
                torch.zeros_like(slice_img_x),
                (slice_img_x + 1.) * self.length / 2.
            ], dim=-1)
        else:
            img_local_pts = torch.stack([
                slice_img_x * self.width / 2.,
                torch.zeros_like(slice_img_x),
                (slice_img_y + 1.) * self.length / 2.
            ], dim=-1)
        self.img_pts = (self.rot * img_local_pts[..., None, :]).sum(-1) + self.origin

    def project(self, pts: torch.Tensor):
        local_pts = (self.rot.t() * (pts - self.origin)[..., None, :]).sum(dim=-1)
        if self.vertical:
            return torch.stack([
                local_pts[2] / self.length * self.resolution[0],
                (local_pts[0] / self.width + .5) * self.resolution[1]
            ], dim=-1)
        else:
            return torch.stack([
                (local_pts[0] / self.width + .5) * self.resolution[0],
                local_pts[2] / self.length * self.resolution[1]
            ], dim=-1)


def render_slice(camera: Camera, rays_o: torch.Tensor, rays_d: torch.Tensor, slice_x: int,
                 slice_y: int, far: float, pix_width: int, pix_length: int, vertical: bool,
                 perspective_warp: bool):
    if vertical:
        rays_o = rays_o[:, slice_x]
        rays_d = rays_d[:, slice_x]
    else:
        rays_o = rays_o[slice_y]
        rays_d = rays_d[slice_y]

    if perspective_warp:
        if vertical:
            u = rays_o.new_full((pix_width,), slice_x + .5)
            v = torch.linspace(.5, camera.intr.H + .5, pix_width, device=rays_o.device)
        else:
            u = torch.linspace(.5, camera.intr.W + .5, pix_width, device=rays_o.device)
            v = rays_o.new_full((pix_width,), slice_y + .5)
        rays_d = nn_f.normalize(camera.world_transform.rotate(camera.intr.lift(u, v)[:, :3]),
                                dim=-1)
        rays_o = camera.world_transform.translation().expand_as(rays_d)
        depths = torch.linspace(0., far, pix_length, device=rays_o.device)
        slice_img_pts = rays_o[:, None] + rays_d[:, None] * depths[:, None]
        if not vertical:
            slice_img_pts = slice_img_pts.permute(1, 0, 2)
    else:
        slice_plane = SlicePlane(rays_o, rays_d, far, pix_width, pix_length, vertical)
        slice_img_pts = slice_plane.img_pts

    slice_img_pts_in_obj = obj.world_transform.forward(slice_img_pts, inv=True)
    slice_img_pts_in_obj = obj_space.normalize_coords(slice_img_pts_in_obj)

    slice_pts_mask = torch.logical_and(slice_img_pts_in_obj >= -1.,
                                       slice_img_pts_in_obj <= 1.).all(-1)
    slice_pts_occ_mask = obj_occ.query(slice_img_pts_in_obj)
    slice_pts_occ_mask &= slice_pts_mask
    d = obj.model.query_sdf(slice_img_pts_in_obj[slice_pts_mask]).to(torch.float)
    slice_colors = color_sdf(d)
    slice_img = torch.zeros_like(slice_img_pts_in_obj)
    slice_img[slice_pts_mask, :] = slice_colors
    slice_img[~slice_pts_occ_mask] *= 0.3
    slice_img = slice_img.clip(0., 1.)
    slice_fig = img_to_fig(slice_img)

    if perspective_warp:
        if vertical:
            slice_fig.add_shape(type="line", xref="x", yref="y",
                                x0=0, y0=(slice_y + .5) / camera.intr.H * pix_width,
                                x1=pix_length - 1, y1=(slice_y + .5) / camera.intr.H * pix_width,
                                line={'color': 'LightSeaGreen', 'width': 1})
        else:
            slice_fig.add_shape(type="line", xref="x", yref="y",
                                x0=(slice_x + .5) / camera.intr.W * pix_width, y0=0,
                                x1=(slice_x + .5) / camera.intr.W * pix_width, y1=pix_length - 1,
                                line={'color': 'LightSeaGreen', 'width': 1})
    else:
        origin_pix = slice_plane.project(rays_o[0])
        leftmost_pix = slice_plane.project(rays_o[0] + rays_d[0] * far)
        rightmost_pix = slice_plane.project(rays_o[0] + rays_d[-1] * far)
        slice_ray_pix = slice_plane.project(
            rays_o[0] + rays_d[slice_y if vertical else slice_x] * far)
        slice_fig.add_shape(type="line", xref="x", yref="y",
                            x0=origin_pix[0].item(), y0=origin_pix[1].item(),
                            x1=leftmost_pix[0].item(), y1=leftmost_pix[1].item(),
                            line={'color': 'Crimson', 'width': 1})
        slice_fig.add_shape(type="line", xref="x", yref="y",
                            x0=origin_pix[0].item(), y0=origin_pix[1].item(),
                            x1=rightmost_pix[0].item(), y1=rightmost_pix[1].item(),
                            line={'color': 'Crimson', 'width': 1})
        slice_fig.add_shape(type="line", xref="x", yref="y",
                            x0=origin_pix[0].item(), y0=origin_pix[1].item(),
                            x1=slice_ray_pix[0].item(), y1=slice_ray_pix[1].item(),
                            line={'color': 'LightSeaGreen', 'width': 1})
    return slice_fig


# Variables
CAMERAS = [
    "camera_FRONT",
    "camera_FRONT_LEFT",
    "camera_FRONT_RIGHT"
]


def create_app():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    lotd_config_spec = {
        "hash_only": {"type": "bool", "label": "Hash only"},
        "prefetch": {"type": "bool", "label": "Prefetch"},
        "permute_dydx": {"type": "bool", "label": "Permute dydx"},
        "calc_dLdx_method": {"type": "int", "label": "Calc dLdx method"},
        "enable_profile": {"type": "bool", "label": "Enable profile"},
    }

    lotd_config_controls = [
        dbc.Row([
            dbc.Label(val["label"], width=4),
            dbc.Col(
                dbc.Switch(id={"type": "lotd-config", "id": key},
                           value=lotd.global_configs[key]) if val["type"] == "bool"
                else dbc.Input(id={"type": "lotd-config", "id": key}, type="number",
                               value=lotd.global_configs[key])
            )
        ], className="mb-3")
        for key, val in lotd_config_spec.items()
    ]

    controls = [
        dbc.Col([
            dbc.Label("Camera", width="auto"),
            dbc.Select(
                id="camera",
                options=[
                    {"label": unsnake(s.replace("camera_", "")), "value": s}
                    for s in CAMERAS
                ],
                value=CAMERAS[0],
            )
        ], lg=2, md=4),
        dbc.Col([
            dbc.Label("Frame", width="auto"),
            dbc.Row([
                dbc.Col(dcc.Slider(0, len(scene) - 1, 1, value=0, id='progression',
                                   marks={0: "0", len(scene) - 1: f"{len(scene) - 1}"},
                                   tooltip={"placement": "bottom", "always_visible": False})),
                dbc.Col(dbc.Input(id="frame_index", type="number", min=0, max=len(scene) - 1, value=0),
                        width=3),
            ], align="center")
        ], lg=4, md=8),
        dbc.Col([
            dbc.Label("Slice", width="auto"),
            dbc.Row(
                dbc.Col(dbc.Checkbox(id="perspective-warp", label="Perspective Warp", value=False))
            )
        ], lg=2, md=4),
    ]
    render_cfg_controls = [
        html.P("Volume Rendering", className="lead"),
        html.Hr(),
        dbc.Row([
            dbc.Label("March Step Size", html_for="march-step-size", width=4),
            dbc.Col(dcc.Slider(0.01, 0.4, 0.01,
                               value=fg_query_cfg["query_param"]["march_cfg"]["step_size"],
                               id="march-step-size",
                               marks={i: str(i) for i in [0.01, 0.1, 0.2, 0.3, 0.4]},
                               tooltip={"placement": "bottom", "always_visible": True})),
        ], className="mb-3",),
        dbc.Row([
            dbc.Label("Coarse", html_for="coarse-samples", width=4),
            dbc.Col(dcc.Slider(0, 256, 16,
                               value=fg_query_cfg["query_param"]["num_coarse"],
                               id="coarse-samples",
                               marks={i: str(i) for i in [0, 64, 128, 256]},
                               tooltip={"placement": "bottom", "always_visible": True})),
        ], className="mb-3",),
        dbc.Row([
            dbc.Label("Fine", html_for="up-samples", width=4),
            dbc.Col(dbc.Input(id="up-samples",
                              value=",".join([str(val)
                                             for val in fg_query_cfg["query_param"]["num_fine"]])
                                    if isinstance(fg_query_cfg["query_param"]["num_fine"], list)
                                    else fg_query_cfg["query_param"]["num_fine"]),
                    width=8),
        ], className="mb-3",),
        dbc.Row([
            dbc.Label("Forward 1/s", html_for="forward-inv-s", width=4),
            dbc.Col(dcc.Slider(800, 60000, 400, value=fg_query_cfg["forward_inv_s"],
                               id="forward-inv-s",
                               marks={i: str(i) for i in [100, 7500, 15000, 30000, 60000]},
                               tooltip={"placement": "bottom", "always_visible": True})),
        ], className="mb-3",),
        dbc.Row([
            dbc.Label("Upsample 1/s", html_for="upsample-inv-s", width=4),
            dbc.Col(dcc.Slider(32, 256, 32, value=fg_query_cfg["query_param"]["upsample_inv_s"],
                               id="upsample-inv-s",
                               marks={i: str(i) for i in [32, 64, 128, 256]},
                               tooltip={"placement": "bottom", "always_visible": True})),
        ], className="mb-3",),
        dbc.Row([
            dbc.Label("Upsample 1/s Factors", html_for="upsample-inv-s-factors", width=4),
            dbc.Col(dbc.Input(id="upsample-inv-s-factors",
                              value=",".join([str(val) for val in fg_query_cfg["query_param"]["upsample_inv_s_factors"]])),
                    width=8),
        ], className="mb-3",),
        html.P("Sphere Tracing", className="lead"),
        html.Hr(),
        dbc.Row([
            dbc.Label("Distance Scale", html_for="spheretrace-distance-scale", width=4),
            dbc.Col(dbc.Input(id="spheretrace-distance-scale",
                    value=sphere_trace_cfg["distance_scale"]), width=8),
        ], className="mb-3",),
        dbc.Row([
            dbc.Label("Min Advance Step", html_for="spheretrace-min-step", width=4),
            dbc.Col(dcc.Slider(0.001, 0.4, 0.001,
                               value=sphere_trace_cfg["min_step"],
                               id="spheretrace-min-step",
                               marks={i: str(i) for i in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4]},
                               tooltip={"placement": "bottom", "always_visible": True})),
        ], className="mb-3",),
        dbc.Row([
            dbc.Label("Hit Thres.", html_for="spheretrace-hit-threshold", width=4),
            dbc.Col(dcc.Slider(0.0005, 0.01, 0.0005,
                               value=sphere_trace_cfg["hit_threshold"],
                               id="spheretrace-hit-threshold",
                               marks={i: str(i) for i in [0.0005, 0.001, 0.005, 0.01]},
                               tooltip={"placement": "bottom", "always_visible": True})),
        ], className="mb-3",),
        dbc.Row([
            dbc.Label("Max Iterations", html_for="spheretrace-max-march-iters", width=4),
            dbc.Col(dcc.Slider(10, 500, 1,
                               value=sphere_trace_cfg["max_march_iters"],
                               id="spheretrace-max-march-iters",
                               marks={i: str(i) for i in [10, 50, 100, 200, 500]},
                               tooltip={"placement": "bottom", "always_visible": True})),
        ], className="mb-3",),
        dbc.Row([
            dbc.Label("Drop Alives (%)", html_for="spheretrace-drop-alive-rate", width=4),
            dbc.Col(dcc.Slider(0, 1, 0.01,
                               value=sphere_trace_cfg["drop_alive_rate"] * 100.,
                               id="spheretrace-drop-alive-rate",
                               marks={i: str(i) for i in [0, 0.25, 0.5, 0.75, 1]},
                               tooltip={"placement": "bottom", "always_visible": True})),
        ], className="mb-3",),
        dbc.Row([
            dbc.Label("Tail Sample Thres.", html_for="spheretrace-tail-sample-threshold", width=4),
            dbc.Col(dbc.Input(id="spheretrace-tail-sample-threshold",
                    value=sphere_trace_cfg["tail_sample_threshold"]), width=8),
        ], className="mb-3",),
        html.P("Lod Module", className="lead"),
        html.Hr(),
        *lotd_config_controls,
        dbc.Row(dbc.Spinner(dbc.Col(dbc.Button("Apply", color="primary", id="apply-render-config"),
                                    className="d-grid gap-2")))
    ]

    profile_grid_cfg = dict(
        columnDefs=[
            # we're using the auto group column by default!
            {"field": "n_calls", "headerName": "Calls", "type": "numericColumn", "width": 90},
            {
                "headerName": "Dur.(ms)",
                "field": "device_duration.sum",
                "type": "numericColumn",
                "width": 110
            },
            {
                "headerName": "%",
                "field": "device_duration.ratio",
                "cellRenderer": 'agSparklineCellRenderer',
                "cellRendererParams": {
                    "sparklineOptions": {
                        "type": 'bar',
                        "fill": '#5470c6',
                        "stroke": '#91cc75',
                        "highlightStyle": {"fill": '#fac858'},
                        "valueAxisDomain": [0, 100],
                        "paddingOuter": 0,
                        "padding": {"top": 3, "bottom": 3},
                        "axis": {"strokeWidth": 0},
                    }
                },
                "width": 60
            },
        ],
        defaultColDef={
            "resizable": True,
        },
        dashGridOptions={
            "autoGroupColumnDef": {
                "headerName": "Node",
                "minWidth": 300,
                "flex": 1,
                "cellRendererParams": {
                    "suppressCount": True,
                },
            },
            "groupDefaultExpanded": -1,
            "getDataPath": {"function": "getDataPath(params)"},
            "treeData": True,
        },
        enableEnterpriseModules=True,
    )
    profile_grid = dag.AgGrid(id="profile-grid", **profile_grid_cfg)
    sp_profile_grid = dag.AgGrid(id="sp-profile-grid", **profile_grid_cfg)

    app.layout = dbc.Container(
        [
            Header("NR3D Rendering Inspector", app),
            html.Br(),
            dbc.Card(dbc.Row(controls, justify="start"), body=True),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Render Config"),
                        dbc.CardBody(render_cfg_controls)
                    ])
                ], lg=3, md=5),
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Rendered"),
                            dbc.CardBody([
                                dbc.Select(
                                    id="rendered-channel",
                                    options=[
                                        {"label": "RGB", "value": "rgb"},
                                        {"label": "Normal", "value": "normal"}
                                    ],
                                    value="rgb",
                                ),
                                dcc.Graph(id="graph-rgb")
                            ])
                        ]), lg=4),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Slice Vertical"),
                            dbc.CardBody(dcc.Graph(id="graph-slice-ver"))
                        ]), lg=8),
                    ]),
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Slice Horizontal"),
                            dbc.CardBody(dcc.Graph(id="graph-slice-hor"))
                        ]), lg=4),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Ray Inspector"),
                            dbc.CardBody(dcc.Graph(id="graph-ray-plot"))
                        ]), lg=8),
                    ]),
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Sphere Trace RGB"),
                            dbc.CardBody(dcc.Graph(id="graph-sp-rgb"))
                        ]), lg=4),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Sphere Trace Inspector"),
                            dbc.CardBody(dcc.Graph(id="graph-sp-plot"))
                        ]), lg=8),
                    ]),
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Volume Render Performance"),
                            dbc.CardBody(profile_grid)
                        ]), lg=6),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Sphere Trace Performance"),
                            dbc.CardBody(sp_profile_grid)
                        ]), lg=6),
                    ])
                ], lg=9, md=7),
            ]),
            dcc.Store(id='local', storage_type='local'),
        ],
        fluid=True,
    )

    @callback(
        Output("frame_index", "value"),
        Input("progression", "value"),
    )
    def update_frame_inputbox(progression_value):
        return int(progression_value)

    @callback(
        Output("graph-rgb", 'figure'),
        Output("graph-sp-rgb", 'figure'),
        Output("graph-slice-hor", "figure"),
        Output("graph-slice-ver", "figure"),
        Output("graph-ray-plot", "figure"),
        Output("graph-sp-plot", "figure"),
        Output("progression", "value"),
        Output("profile-grid", "rowData"),
        Output("sp-profile-grid", "rowData"),
        Output("apply-render-config", "title"),
        Output("local", "data"),
        Input("frame_index", "value"),
        Input("camera", "value"),
        Input("perspective-warp", "value"),
        Input("graph-rgb", "clickData"),
        Input("graph-sp-rgb", "clickData"),
        Input("apply-render-config", "n_clicks"),
        Input("rendered-channel", "value"),
        State("coarse-samples", "value"),
        State("up-samples", "value"),
        State("forward-inv-s", "value"),
        State("upsample-inv-s", "value"),
        State("upsample-inv-s-factors", "value"),
        State("march-step-size", "value"),
        State("spheretrace-distance-scale", "value"),
        State("spheretrace-min-step", "value"),
        State("spheretrace-max-march-iters", "value"),
        State("spheretrace-drop-alive-rate", "value"),
        State("spheretrace-tail-sample-threshold", "value"),
        State("spheretrace-hit-threshold", "value"),
        State({"type": "lotd-config", "id": ALL}, "value"),
        State("local", "data"),
    )
    @torch.no_grad()
    def update_graphs(frame_index, camera_id, perspective_warp,
                      rgb_graph_click, sp_rgb_graph_click,
                      apply_render_config_click, show_channel, coarse_samples, up_samples,
                      forward_inv_s, upsample_inv_s, upsample_inv_s_factors, march_step_size,
                      spheretrace_distance_scale, spheretrace_min_step, spheretrace_max_march_iters,
                      spheretrace_drop_alive_rate, tail_sample_threshold, hit_threshold,
                      lotd_configs, local_data):
        global rgb_fig, sp_rgb_fig, x, y

        frame_index = min(max(int(frame_index), 0), len(scene))
        scene.slice_at(frame_index)
        camera = scene.get_cameras()[camera_id]
        camera.intr.set_downscale(args.downscale)
        rays_o, rays_d, rays_ts = camera.get_all_rays(return_ts=True)
        rays_ts = rays_ts.reshape(camera.intr.H, camera.intr.W, -1)
        rays_o = rays_o.reshape(camera.intr.H, camera.intr.W, -1)
        rays_d = rays_d.reshape(camera.intr.H, camera.intr.W, -1)

        lotd_configs = dict(zip(lotd_config_spec, lotd_configs))
        lotd.configure(**lotd_configs)

        if not ctx.triggered or ctx.triggered_id == "frame_index" or ctx.triggered_id == "camera" \
                or ctx.triggered_id == "lotd-hash-only" or ctx.triggered_id == "lotd-enable-profile" \
                or ctx.triggered_id == "rendered-channel":
            rgb_fig = None
            sp_rgb_fig = None
        elif ctx.triggered_id == "apply-render-config":
            fg_query_cfg["forward_inv_s"] = int(forward_inv_s)
            fg_query_cfg["query_param"]["num_coarse"] = int(coarse_samples)
            fg_query_cfg["query_param"]["num_fine"] = [int(val) for val in up_samples.split(",")] \
                if "," in up_samples else int(up_samples)
            fg_query_cfg["query_param"]["upsample_inv_s"] = int(upsample_inv_s)
            fg_query_cfg["query_param"]["upsample_inv_s_factors"] = \
                [float(val) for val in upsample_inv_s_factors.split(",")]
            fg_query_cfg["query_param"]["march_cfg"]["step_size"] = march_step_size
            sphere_trace_cfg["distance_scale"] = float(spheretrace_distance_scale)
            sphere_trace_cfg["min_step"] = float(spheretrace_min_step)
            sphere_trace_cfg["max_march_iters"] = int(spheretrace_max_march_iters)
            sphere_trace_cfg["drop_alive_rate"] = float(spheretrace_drop_alive_rate) / 100.
            sphere_trace_cfg["tail_sample_threshold"] = int(tail_sample_threshold)
            sphere_trace_cfg["hit_threshold"] = float(hit_threshold)
            rgb_fig = None
            sp_rgb_fig = None
        elif ctx.triggered_id == "graph-rgb":
            x = rgb_graph_click['points'][0]['x']
            y = rgb_graph_click['points'][0]['y']
            local_data = {"x": x, "y": y}
        elif ctx.triggered_id == "graph-sp-rgb":
            x = sp_rgb_graph_click['points'][0]['x']
            y = sp_rgb_graph_click['points'][0]['y']
            local_data = {"x": x, "y": y}

        if x is None:
            if not local_data:
                x, y = camera.intr.W // 2, camera.intr.H // 2
                local_data = {"x": x, "y": y}
            else:
                x, y = local_data["x"], local_data["y"]

        profile_grid_data = dash.no_update
        sp_profile_grid_data = dash.no_update
        if rgb_fig is None:
            profiler = Profiler(0, 1).enable()
            rgb_fig = img_to_fig(render(camera, return_channel=show_channel))
            profile_grid_data = profiler.get_result().get_statistic("device_duration") \
                .get_raw_data(sort_by="device_duration")
            for item in profile_grid_data:
                item["device_duration"]["sum"] = int(item["device_duration"]["sum"] * 100) / 100
                item["device_duration"]["ratio"] = [
                    int(item["device_duration"]["ratio"] * 10000) / 100]
        if sp_rgb_fig is None:
            profiler = Profiler(0, 1).enable()
            sp_rgb_fig = img_to_fig(render(camera, True, return_channel=show_channel))
            sp_profile_grid_data = profiler.get_result().get_statistic("device_duration") \
                .get_raw_data(sort_by="device_duration")
            for item in sp_profile_grid_data:
                item["device_duration"]["sum"] = int(item["device_duration"]["sum"] * 100) / 100
                item["device_duration"]["ratio"] = [
                    int(item["device_duration"]["ratio"] * 10000) / 100]

        # print(f"Update inspect slice at {x},{y}")
        rgb_fig.update_shapes({'visible': False})
        rgb_fig.add_shape(type="line", xref="x", yref="y",
                          x0=x, y0=0,
                          x1=x, y1=camera.intr.H,
                          line={'color': 'LightSeaGreen', 'width': 1})
        rgb_fig.add_shape(type="line", xref="x", yref="y",
                          x0=0, y0=y,
                          x1=camera.intr.W, y1=y,
                          line={'color': 'LightSeaGreen', 'width': 1})

        sp_rgb_fig.update_shapes({'visible': False})
        sp_rgb_fig.add_shape(type="line", xref="x", yref="y",
                             x0=x, y0=0,
                             x1=x, y1=camera.intr.H,
                             line={'color': 'LightSeaGreen', 'width': 1})
        sp_rgb_fig.add_shape(type="line", xref="x", yref="y",
                             x0=0, y0=y,
                             x1=camera.intr.W, y1=y,
                             line={'color': 'LightSeaGreen', 'width': 1})

        slice_hor_fig = render_slice(camera, rays_o, rays_d, rays_ts, x, y, camera.far,
                                     pix_width=2000, pix_length=2000, vertical=False,
                                     perspective_warp=perspective_warp)
        slice_ver_fig = render_slice(camera, rays_o, rays_d, rays_ts, x, y, camera.far,
                                     pix_width=2000, pix_length=4000, vertical=True,
                                     perspective_warp=perspective_warp)

        ray_plot_fig = go.Figure()
        plot_sdf(ray_plot_fig, rays_o[y, x], rays_d[y, x], rays_ts[y, x], camera.near, camera.far, 1000)
        plot_volume_render_samples(ray_plot_fig, camera, rays_o[y, x], rays_d[y, x], rays_ts[y, x])

        sp_plot_fig = go.Figure()
        plot_sdf(sp_plot_fig, rays_o[y, x], rays_d[y, x], rays_ts[y, x], camera.near, camera.far, 1000, True, True)
        plot_sphere_trace_samples(sp_plot_fig, camera, rays_o[y, x], rays_d[y, x], rays_ts[y, x])

        return rgb_fig, sp_rgb_fig, slice_hor_fig, slice_ver_fig, ray_plot_fig, sp_plot_fig, \
            frame_index, profile_grid_data, sp_profile_grid_data, dash.no_update, local_data
    return app


if __name__ == "__main__":
    from nr3d_lib.config import BaseConfig
    bc = BaseConfig()
    bc.parser.add_argument("--downscale", type=float, default=1.0)
    bc.parser.add_argument("--rayschunk", type=int, default=65536)
    bc.parser.add_argument("--rayschunk_for_bg", type=int, default=2**16)
    bc.parser.add_argument("--regenerate_occgrid", type=int)
    args = bc.parse(print_config=False)

    scene_bank, asset_bank, _ = load_scenes_and_assets(**args, device=device)
    scene = scene_bank[0]
    obj = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]
    obj_model = obj.model
    obj_space: AABBSpace = obj_model.space

    # Sphere tracing config: for waymo street scene
    if scene.main_class_name == "Street":
        sphere_trace_cfg = ConfigDict(
            distance_scale=40.,
            min_step=.2,
            max_march_iters=500,
            drop_alive_rate=0.,
            tail_sample_threshold=20000,
            tail_sample_step_size=None,
            hit_threshold=.01
        )

    # Sphere tracing config: for single-object scene
    if scene.main_class_name == 'Main':
        sphere_trace_cfg = ConfigDict(
            distance_scale=1.,
            min_step=.002,
            max_march_iters=500,
            drop_alive_rate=0.,
            tail_sample_threshold=10000,
            tail_sample_step_size=None,
            hit_threshold=1e-4
        )

    CAMERAS = [cam.id for cam in scene.get_cameras(False)]

    if args.regenerate_occgrid:
        print("Regenerate Occgrid...")
        print("Original resolution: ", obj_model.accel.occ.resolution.tolist())
        resolution = (obj.model.accel.occ.resolution * args.regenerate_occgrid).tolist()
        obj_model.accel.occ = OccGridEma(
            resolution, occ_val_fn_cfg=ConfigDict(type='raw_sdf'), occ_thre=1. - 0.002,
            init_cfg=ConfigDict(num_steps=256, num_pts=2**24, mode='from_net'))
        obj_model.accel.init(obj_model.query_sdf)

    obj_occ = obj_model.accel.occ
    obj.model.training_before_per_step(args.training.num_iters)  # Set to last state

    # ---------------------------------------------
    # ------------     Renderer     ---------------
    # ---------------------------------------------
    renderer = SingleVolumeRenderer(args.renderer)
    renderer.populate(asset_bank)
    renderer.eval()

    renderer.config.rayschunk = args.rayschunk
    renderer.config.with_normal = False
    for scene in scene_bank:
        # NOTE: When training, set all observer's near&far to a larger value
        for obs in scene.get_observers(False):
            obs.near = renderer.config.near
            obs.far = renderer.config.far or 5.0

    fg_query_cfg = obj.model.ray_query_cfg
    fg_query_cfg["forward_inv_s"] = obj.model.forward_inv_s().item()

    app = create_app()
    app.run(debug=False, port="8051")
