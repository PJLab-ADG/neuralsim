- [Implemented methods](#implemented-methods)
- [General usage](#general-usage)

## Implemented methods

### Object-centric

| Methods                                                      | Official / Un-official | Get started                        | Notes, major difference from paper, etc.                     |
| ------------------------------------------------------------ | ---------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| [NeuS](https://lingjie0206.github.io/papers/NeuS/) in minutes | Un-official            | [readme](../docs/methods/neus_in_10_minutes.md) | - support object-centric datasets as well as <u>indoor</u> datasets<br />- fast and stable convergence without needing mask<br />- support using [NGP](https://github.com/NVlabs/instant-ngp) / [LoTD](https://github.com/pjlab-ADG/nr3d_lib#pushpin-lotd-levels-of-tensorial-decomposition-) or MLPs as cr&dv representations<br />- large pixel batch size (4096) & pixel error maps |
| [NGP](https://github.com/NVlabs/instant-ngp) | Un-official |  | - support using [NGP](https://github.com/NVlabs/instant-ngp) / [LoTD](https://github.com/pjlab-ADG/nr3d_lib#pushpin-lotd-levels-of-tensorial-decomposition-) or MLPs as representations |

### Street-view

| Methods                                                  | Official / Un-official | Get started                   | Notes, major difference from paper, etc.                     |
| -------------------------------------------------------- | ---------------------- | --------------------------------------- | ------------------------------------------------------------ |
| [StreetSurf](https://ventusff.github.io/streetsurf_web/) | Official               | [readme](../docs/methods/streetsurf.md) | - LiDAR loss improved (using L1 and discarding outliers)     |
| [NGP](https://github.com/NVlabs/instant-ngp) with LiDAR  | Un-official            | [readme](../docs/methods/ngp_lidar.md)  | - using [Urban-NeRF](https://urban-radiance-fields.github.io/)'s LiDAR loss |

## General usage

- [Dataset preparation](#dataset-preparation)
- [Training](#training)
  - [\> Start a fresh new training](#-start-a-fresh-new-training)
  - [\> Resume a previous experiment](#-resume-a-previous-experiment)
  - [\> Rich training logs](#-rich-training-logs)
  - [\> Optional DDP training](#-optional-ddp-training)
  - [\> Debug training errors](#-debug-training-errors)
- [Rendering](#rendering)
  - [\> Replay](#-replay)
  - [\> NVS](#-nvs)
  - [\> With mesh visualization](#-with-mesh-visualization)
- [Appearance evaluation](#appearance-evaluation)
- [LiDAR simulation](#lidar-simulation)
  - [\> Simulate a single LiDAR](#-simulate-a-single-lidar)
  - [\> Simulate a demo of a list of LiDAR models](#-simulate-a-demo-of-a-list-of-lidar-models)
- [LiDAR evaluation](#lidar-evaluation)
- [Mesh extraction](#mesh-extraction)
- [Occupancy grid extraction](#occupancy-grid-extraction)
  - [\> Format of the extracted occupancy grid](#-format-of-the-extracted-occupancy-grid)


### Dataset preparation

- [[readme]](../dataio/dtu/README.md) :arrow_left: [NeuS](https://github.com/Totoro97/NeuS)'s version of [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36) / [BlendedMVS](https://github.com/YoYo000/BlendedMVS) Dataset
- [[readme]](../dataio/bmvs/README.md) :arrow_left: [BlendedMVS](https://github.com/YoYo000/BlendedMVS) Dataset
- [[readme]](../dataio/monosdf/README.md) :arrow_left: [MonoSDF](https://github.com/autonomousvision/monosdf)'s version of [Replica](https://github.com/facebookresearch/Replica-Dataset) / [scannet](http://www.scan-net.org/) Dataset
- [[readme]](../dataio/autonomous_driving/waymo/README.md) :arrow_left: [Waymo Open Dataset - Perception](https://waymo.com/open/data/perception/)

---

NOTE:

- :running: You can combine multiple subtasks listed below and automatically execute them one by one with [run.py](tools/run.py) . For example:

  - ```shell
    python code_single/tools/run.py train,eval,eval_lidar,extract_mesh \
    --config code_single/configs/xxx.yaml \
    --eval.downscale=2 --eval_lidar.lidar_id=lidar_TOP \
    --extract_mesh.to_world --extract_mesh.res=0.1
    ```
  - `--config` or `--resume_dir` are common args shared across all subtasks.

- :pushpin: All the instructions below assume you have already `cd` into `/path/to/neuralsim` .

### Training

#### > Start a fresh new training

```shell
python code_single/tools/train.py --config code_single/configs/xxx.yaml
```

:gear: You can specify temporary configs via command line args like `--aaa.bbb=ccc`, which will temporarily modify the `aaa:bbb` field in `xxx.yaml` in this run. For more details on how the command line and yaml configuration work, please refer to [this doc](https://github.com/PJLab-ADG/nr3d_lib/blob/main/docs/config.md) .

#### > Resume a previous experiment

```shell
python code_single/tools/train.py --resume_dir /path/to/logs/xxx
```

You can also resume a experiment if everything is not changed in the original config yaml by directly specifying `--config code_single/configs/xxx.yaml` . 

#### > Rich training logs

We provide rich logging information with tensorboard. 

Check them out by

```shell
tensorboard --logdir /path/to/logs/xxx
```

The logging frequency of scalars is controlled by `training:i_log` field. (how many iterations per log entry).

The logging frequency of images (visualization or renderings) is controlled by `training:i_val` field.

#### > Optional DDP training

:arrow_right: **Single node multi GPUs**

Taking an example of a single machine with 4 GPUs:

:pushpin: NOTE: You only need to add the `--ddp` option to the command line arguments of `train.py`.

```shell
python -m torch.distributed.launch --nproc_per_node=4 \
code_single/tools/train.py \
--config code_single/configs/waymo/streetsurf/withmask_withlidar.230814.yaml \
--ddp
```

In the above example, if everything works properly, you will see the following message printed four times with differen ranks in the logs:

<details>

```
=> Enter init_process_group(): 
	=> rank=0
	=> world_size=4
	=> local_rank=0
	=> master_addr=127.0.0.1
	=> master_port=29500
...
=> Done init Env @ DDP: 
	=> device_ids set to [0]
	=> rank=0
	=> world_size=4
	=> local_rank=0
	=> master_addr=127.0.0.1
	=> master_port=29500
...
```

</details>

:arrow_right: **Multi nodes multi GPUs**

Taking an example of a 2 nodes with 4 GPUs each (i.e. 8 GPUs in total):

:pushpin: NOTE: You only need to add the `--ddp` option to the command line arguments of `train.py`.

```shell
python -m torch.distributed.launch --nnodes=$WORLD_SIZE --nproc_per_node=4 \
--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$RANK \
code_single/tools/train.py \
--config code_single/configs/waymo/streetsurf/withmask_withlidar.230814.yaml \
--ddp 
```

In the above example, if everything works properly, you will see the following message printed four times with differen ranks in the logs of **the master node**:

<details>

```
=> Enter init_process_group(): 
	=> rank=0
	=> world_size=8
	=> local_rank=0
	=> master_addr=dlcfevx8ltikuljg-master-0
	=> master_port=23456
...
=> Done init Env @ DDP: 
	=> device_ids set to [0]
	=> rank=0
	=> world_size=8
	=> local_rank=0
	=> master_addr=dlcfevx8ltikuljg-master-0
	=> master_port=23456
...
```

As for the worker node's print logs:

```
=> Enter init_process_group(): 
	=> rank=4
	=> world_size=8
	=> local_rank=0
	=> master_addr=dlcfevx8ltikuljg-master-0
	=> master_port=23456
...
=> Done init Env @ DDP: 
	=> device_ids set to [0]
	=> rank=4
	=> world_size=8
	=> local_rank=0
	=> master_addr=dlcfevx8ltikuljg-master-0
	=> master_port=23456
...
```

</details>

#### > Debug training errors

We also provide a primitive debugging tool for checking gradients. You can try it out by modifying `self.debug_grad=True` in the `Trainer` class. Note that this will significantly slow down training and should be used along with `debugpy` or other tools.

### Rendering

The [tools/render.py](tools/render.py) works in two modes, namely replay or NVS (novel_view_synthesis) mode. Both modes support additional LiDAR simulation or mesh visualization along with rgb, depth and surface normals rendering.

#### > Replay

By default, [tools/render.py](tools/render.py) runs in replay mode, which will render frames between the optionally given `--start_frame` and `--stop_frame` parameter with everything untouched.

```shell
python code_single/tools/render.py --resume_dir /path/to/logs/xxx --downscale=1 \
--assetbank_cfg.Main.model_params.ray_query_cfg.query_param.num_coarse=0
```

NOTE:

- For street-view, rendering full size videos often consumes a lot of time. It is recommended to specify `--downscale=2` or larger values.
- Usually, ignoring `num_coarse` samples will not significantly affect the results and will speed up rendering. 
  - For StreetSurf, simply add `--assetbank_cfg.Street.model_params.ray_query_cfg.query_param.num_coarse=0`
  - For other single object datasets, simply add `--assetbank_cfg.Main.model_params.ray_query_cfg.query_param.num_coarse=0`

- Many other options can be specified while rendering, including `--no_sky`, `--only_cr`,  `--fps`, `--rayschunk` etc. Check out [tools/render.py](tools/render.py) for more details.

#### > NVS

By giving `--nvs_path=...` etc. to specify the type of the novel camera trajectory and other configs, [tools/render.py](tools/render.py) runs in NVS mode.

`--nvs_node_id` is used to specify the scene graph node whose trajectory you wish to manipulate. Typically, for single-object datasets, this node is `camera`. For street-view datasets, it's `ego_car`.

:arrow_right: **Example for single-object NVS**

```shell
python code_single/tools/render.py --resume_dir logs/bmvs/5c0d13 \
--nvs_path=spherical_spiral --nvs_node_id=camera --nvs_param=48,29,54 \
--nvs_num_frames=120 --downscale=1 \
--assetbank_cfg.Main.model_params.ray_query_cfg.query_param.num_coarse=0
```

:arrow_right: **Example for street-view NVS**

NOTE: `--start_frame` and `--stop_frame` in this case specifies the reference frames for the camera path creation method. The real length of the NVS path is specified by `--nvs_num_frames`.

```shell
python code_single/tools/render.py --resume_dir logs/streetsurf/seg100613 \
--nvs_path=street_view --nvs_node_id=ego_car --nvs_param=2.0,1.0,3.0,0.0,2.0,-2.0 \
--nvs_num_frames=120 --start_frame=80 --stop_frame=160 --downscale=4 \
--assetbank_cfg.Street.model_params.ray_query_cfg.query_param.num_coarse=0
```

#### > With mesh visualization

<details>
<summary>An example (click to expand)</summary>

<img src="https://github.com/PJLab-ADG/neuralsim/assets/25529198/7c1767c6-6eae-41c4-a60d-2eb74ff4c4c9" alt="seg100613_ds=4_withmesh" width="640">

</details>

To visualize a specific mesh from the perspective of the original cameras when rendering, additionally specify `--render_mesh=xxx.ply` :

```shell
python code_single/tools/render.py --resume_dir /path/to/logs/xxx \
--downscale=4 --render_mesh /path/to/logs/xxx/meshes/xxx.ply \
--render_mesh_transform=identity \
--assetbank_cfg.Main.model_params.ray_query_cfg.query_param.num_coarse=0
```

NOTE: If the input mesh is already in world coordinates (e.g. `--to_world` is specified when extracting mesh), `--render_mesh_transform` should just be `identity`. If the input mesh is in object coordinates, `--render_mesh_transform` should be `to_world`.

### Appearance Evaluation

This is similar to the replay mode in [tools/render.py](tools/render.py), but with additional calculations and limitations for evaluation.

```shell
python code_single/tools/eval.py --resume_dir /path/to/logs/xxx
```

### LiDAR simulation

#### > Simulate a single LiDAR

For example, to simulate the original LiDAR model:

```shell
python code_single/toos/render.py --resume_dir /path/to/logs/xxx \
--no_cam --render_lidar --lidar_model=original_reren --lidar_id=lidar_TOP
```

:arrow_right: A visualization window can be popped up by additionally specifying `--lidar_vis_verbose`.

You can also try this out when rendering in NVS mode.

#### > Simulate a demo of a list of LiDAR models

In addition to the original LiDAR, numerous other real-world LiDAR models can be simulated.

:arrow_right: Below is a script that sequentially simulates a list of LiDAR models:

```shell
bash code_single/tools/demo_lidar_sim.sh /path/to/logs/xxx --lidar_vis_width=1200
```

:arrow_right: A visualization window can be popped up by additionally specifying `--lidar_vis_verbose`.

### LiDAR evaluation

```shell
python code_single/tools/eval_lidar.py --resume_dir /path/to/logs/xxx \
--lidar_id=lidar_TOP --dirname=eval_lidar_TOP
```

:arrow_right: A visualization video like the one in [StreetSurf website](https://ventusff.github.io/streetsurf_web/) can be produced by additionally specifying `--video_backend=vedo`:

```shell
python code_single/tools/eval_lidar.py --resume_dir /path/to/logs/xxx \
--lidar_id=lidar_TOP --dirname=eval_lidar_TOP --video_backend=vedo
```

:arrow_right: A visualization window can be popped up by additionally specifying `--video_verbose`.

### Mesh extraction

To extract mesh of a specific experiment:

:arrow_right: For SDF networks:

```shell
python code_single/tools/extract_mesh.py --resume_dir /path/to/logs/xxx \
--to_world --res=0.1
```

:arrow_right: For NeRF networks: (you can specify other sigma threshold with `--levelset=` )

```shell
python code_single/tools/extract_mesh.py --resume_dir /path/to/logs/xxx \
--to_world --res=0.1 --network_type=nerf --levelset=1.0
```

### Occupancy grid extraction

```shell
python code_single/tools/extract_occgrid.py --resume_dir /path/to/logs/xxx \
--occ_res=0.1
```

A visualization window can be popped up when the extraction is finished by additionally specifying `--verbose` (:warning: Might run out of CPU mem if `occ_res` is small and the resulting resolution is large).

#### > Format of the extracted occupancy grid

We opt to store the actual occupied integer coordinates rather than a full-resolution 3D boolean grid to save space.

The output file is in `.npz` format, containing occupied vertex coordinates and meta information.

Below is a description and example of how to read the file:

```python
import numpy as np
datadict = np.load("xxx.npz", allow_pickle=True)
datadict['occ_corners'] # [N, 3], int16, integer coordinates of the actual occupied grid points, where N represents the number of actual occupied grids
datadict['sidelength'] # [res_x, res_y, res_z], int, integer side lengths allocated in x, y, z directions respectively
datadict['occ_res'] # float, default 0.1, resolution setting when extracting occupied grid, i.e., the side length of each cubic grid
datadict['coord_min'] # [3,], float, world coordinates corresponding to the vertex at the front-left-bottom corner (the vertex with smaller values in x, y, z directions) of the integer coordinate [0,0,0] grid
datadict['coord_offset'] # [3,], float, offset between the world coordinate system definition and the world coordinate system definition of the original data sequence (original_world=current_world+coord_offset)
datadict['meta'] # dict, a dictionary containing the meta information of the scene
datadict['meta']['scene_id'] # str, full name id of the current sequence "segment-xxxxx-with_camera_labels"
datadict['meta']['start_frame'] # int, the start frame defined during the training of the current sequence
datadict['meta']['num_frames'] # int, the total number of frames defined during the training of the current sequence (end frame = start frame + total number of frames)

# To read:
voxel_coords_in_world = datadict['occ_corners'].astype(float) * datadict['occ_res'] + datadict['coord_min']
voxel_coords_in_data_world = voxel_coords_in_world + datadict['coord_offset']
```

