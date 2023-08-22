# [neuralsim] 自动驾驶数据的处理格式标准

[English](autonomous_driving.md)|中文

## Overview

总的来说，最终需要处理形成的数据是以相同的、一致长短的整数帧索引同步存储的图像、lidar数据，以及各物体和自车位姿等元信息。

对于单个序列而言，序列数据大致如下：

```
seq_root_dir
├── images
│   ├── camera_0
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ├── ...
│   ├── camera_1
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ├── ...
│   ├── ...
├── lidars
│   ├── lidar_0
│   │   ├── 00000000.npz
│   │   ├── 00000001.npz
│   │   ├── ...
│   ├── lidar_1
│   │   ├── 00000000.npz
│   │   ├── 00000001.npz
│   │   ├── ...
│   ├── ...
└── scenario.pt
```

其中，pickle 文件 `scenario.pt` 中存储着除了原始传感器数据外的其他所有数据，包括自车的位姿、他车的 tracklets、不同传感器的标定内外参等。

## scenario.pt 格式要求

`scenario.pt` 存储了一个字典，有4个主要的key，`observers`, `objects`, `scene_id`, `metas` 。

- `observers` 代表所有的传感器，这里主要指相机和 lidar，如果有需要的话也可以放入自车 `ego_car` 的信息。

- `objects` 代表所有的其他物体，注意自车不属于其他物体的行列。
- `scene_id` 是场景的字符串 id，主要是方便标识，可以随意指定。
- `metas` 代表其他场景元数据

下面是这个字典的一个范例：

```python
{
    'observers': {  # 所有的传感器，包括 camera 和 lidar
        'camera_0': {
            'id': 'camera_0', 
            'class_name': 'Camera', 
            'n_frames': 200, 
            'data': {...} # 相机的从头到尾的逐帧元数据，具体见相机章节
        }, 
        ..., 
        'lidar_0': {
            'id': 'lidar_0', 
            'class_name': 'RaysLidar', 
            'n_frames': 200, 
            'data': {...} # lidar的从头到尾的逐帧元数据，具体见lidar章节
        }, 
        ..., 
        'ego_car': { # 注：可选，可以没有。
            'id': 'ego_car', 
            'class_name': 'EgoVehicle', 
            'n_frames': 200, 
            'data': {...} # 自车的从头到尾的逐帧元数据
        }
    }, 
    'objects': { # 如果这个序列没有他车/他人/etc.，可以是一个空 dict 
        'obj0': {
            'id': 'obj0', 
            'class_name': 'Vehicle', 
            'segments': # 片段数据列表，每个item是一个字典。对于某个 他车/他人/etc. 可能会有多个可见片段的数据，不过一般只有一个。
            [
                {
                    'start_frame': 12, 
                    'n_frames': 30, 
                    'data': {...} # 在这个片段内的逐帧元数据
                }, 
                {
                    'start_frame': 100, 
                    'n_frames': 40, 
                    'data': {...} # 在这个片段内的逐帧元数据
                }, 
                ... # 更多片段，如果有的话；每个都是一个字典，一个 list 中的 item 
            ]
        }, 
        'obj1': {...}, 
        ...
    }
    'scene_id': 'your_favorite_seq_name', 
    'metas': {
        'num_frames': 198 # int, 整个场景的总帧长。
        'world_offset': world_offset # [3,], np.ndarray, float, 我们所使用的世界坐标系和原始世界坐标系的位置差异
        'up_vec': '+z' # 代表了竖直向上方向在当前坐标系定义下的方向，如 +z 代表 z 轴的正半轴方向。
    }
}
```

### 世界坐标系

自车、所有的传感器、所有的其他物体的姿态矩阵信息都定义在统一的同一个世界坐标系下。

- 目前，我们选取和原始数据集的世界坐标系朝向相同的坐标系，并定义第0帧的自车translation作为坐标系原点
  - 换句话说，第0帧的自车translation即为 我们代码仓 所使用的世界坐标系和原始世界坐标系时间的 offset

  - 这个 offset（3维浮点数numpy向量） 存储在 `scneario.pt` 的 `['metas']['world_offset']` 以备不时之需。

- 我们代码仓支持这个坐标系可以具有任意的朝向。
  - 不过目前的训练过程中，需要获知重力方向的反方向（i.e.竖直向上的方向）在坐标系中的朝向，比如 `+z` 这样的方式，代表了 z 轴的正半轴方向为竖直向上的方向。这个信息存储在 `scneario.pt` 的 `['metas']['up_vec']` 


### 逐帧元数据

一般而言，自车车身上的所有的传感器以及自车本身，在整个序列中都是包含从头到尾每一帧的元数据的；相对的，所观察到的他车/他人等，往往只有一些片段是可见的（这里指总的可见性，包括相机可见和lidar可见）。

因此，`observer` 和 `objects` 的逐帧元数据的存储方式是不同的：

- `observers` 直接在 其 dict 中定义一个 `data` 字典，每个字段直接存储全帧长的矩阵数据信息。
  - 这主要是由于 `observer` 是全时刻存在的。
-  `objects` 首先定义了一个 `segments` 列表 `[]`，然后在列表中放置各个可见片段作为列表的一个个item，每个片段是一个字典 `{}`，包含 `start_frame`，`n_frames` 和 `data` 字段。`data` 字段是个字典，其每个字段存储一个片段帧长的矩阵数据信息。
  - 这主要是由于其他物体往往只在自车的采集周期中出现若干个不全长的片段。

具体的相机、lidar 和其他物体字典 存储如何的逐帧元数据，参见后面的相机和 lidar 的章节。

### 时间同步要求

并不要求严苛的时间同步，只要求一个大概的、泛泛的整数帧，允许不同传感器的采集时间戳有差异；比如同样对于第 00000000 帧，camera_0 和 camera_1 的实际采集时间可能相差若干毫秒。

如果能够提供不同传感器采集时间戳时刻的自车 pose，可以额外提供；如果不能，将依赖重建过程中的 pose refine 过程减小或消除由不精准的时间戳带来的姿态误差。

我们不要求 lidar 和 camera 的整数帧是同步或者对应的，因为 lidar 的监督信号是单独在 lidar 的坐标系下渲染和使用的，并不涉及投影到相机坐标系的过程；但目前暂时要求二者帧数长短相同。

## camera 格式要求

### 图像数据

- 支持不同 camera 的图像具有不同的尺寸。
  - 以 waymo 数据集为例，三个前向的相机 `resolution=[1920x1280]`，两个侧向的相机 `resolution=[1920x886]`

- 目前暂时要求同一 camera 不同帧的图像具有相同的尺寸。
- 图片可以是任意主流的 RGB 图像格式。暂时不支持 HDR 图像。

### 内外参等逐帧元数据

在 `scenario.pt` 的字典中，相机的元数据存储在 `obsevers` 字段的字典里。

一个范例相机元数据如下：

```python
'observers': { 
    'camera_0': {
        'id': 'camera_0', 
        'class_name': 'Camera', 
        'n_frames': 200, 
        'data': { # 相机的从头到尾的逐帧元数据
            'hw': ...,  # [200, 2], np.ndarray, long, 逐帧的图像的高宽
            'intr': ..., # [200, 3, 3], np.ndarray, float, 逐帧的相机的内参（一般每一帧都是一样的）
            'c2w': ..., # [200, 4, 4], np.ndarray, float, openCV 坐标系下的 camera to world 姿态变换矩阵
            'distortion': ..., # [200, ...], np.ndarray, float, 可选的畸变参数 (目前支持标准 openCV 相机和鱼眼相机)
        }
    }, 
    ...
}, 
```

#### > 内参

`intr`  是 3x3 的针孔相机内参矩阵 `[fx, sk, cx; 0, fy, cy; 0, 0, 1]` ，其中 `fx`, `fy`, `cx`, `cy` 单位均为像素。

`hw` 是逐帧的图像高宽，单位均为像素。

`distortion` 是可选的相机畸变参数，目前支持 [OpenCV camera](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) 和 [fisheye camera](https://docs.opencv.org/4.7.0/db/d58/group__calib3d__fisheye.html) 两种定义（fisheye camera 的 GPU 版本正在实现中）

如果采用 [OpenCV camera](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)，合法尺寸为 `[4, 5, 8, 12, 14]`，符号 $(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])$

如果采用 [fisheye camera](https://docs.opencv.org/4.7.0/db/d58/group__calib3d__fisheye.html)，合法尺寸为 `[4]`，符号 $(k_1,k_2,k_3,k_4)$

#### > 相机姿态：`c2w`

`c2w` 定义的是 opencv 坐标系下的 camera 到世界坐标系（已经减去了 `world_offset` 的统一世界坐标系）的姿态变换矩阵。

如果相机没有定义在 openCV 坐标系下，需要利用 opencv 坐标系到你的坐标系的变换矩阵做一次转换。

```python
c2w = c2w_of_yours @ opencv_to_yours
```

这里我们以 waymo 坐标系为例：

```python
"""
    < opencv / colmap convention >                 --->>>   < waymo (camera) convention >
    facing [+z] direction, x right, y downwards    --->>>  facing [+x] direction, z upwards, y left
                z                                          z ↑ 
              ↗                                              |  ↗ x
             /                                               | /
            /                                                |/
            o------> x                              ←--------o
            |                                      y
            |
            |
            ↓ 
            y
"""
# NOTE: This matrix converts Opencv camera's vectors to waymo camera's vectors
opencv_to_waymo = np.eye(4)
opencv_to_waymo[:3 ,:3] = np.array(
    [[0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]])

# NOTE: From waymo world to out world
v2w[:3, 3] -= world_offset

# NOTE: Waymo: extrinsic=[camera to vehicle]
c2w = v2w @ c2v @ opencv_to_waymo
```

其中，`v2w` 是 waymo 数据定义的 vehicle 姿态，即 vehicle 到世界的变换矩阵；`c2v` 是 waymo 数据标注的相机外参（的逆），即相机在自车系下的姿态，或相机到自车的姿态变换矩阵。

因此，上面求得的  `c2w` 矩阵 中首先通过 `opencv_to_waymo` 将一个 opencv 相机坐标系下的向量转换到了 waymo 相机坐标系，然后进一步左乘 `v2w @ c2v` 变换到了 waymo 的世界坐标系下，构成了一个 opencv camera to waymo 的变换矩阵。

**注意：这里的世界指的是 ours 偏置后的世界，注意到 `v2w` 需要减去 `world_offset`**

## Lidar 数据处理的格式要求

这里是我们要求的数据形式中较为特殊的地方：我们需要获知原始的 lidar 光束方向向量和 ranges 测量值，因为我们的算法是将 lidar 作为稀疏深度传感器来使用的，这个做法在包含动态物体的多 SDF 重建中尤其有用。

### 传感器数据

具体而言，每一帧的 lidar 采集结果将是一个通过 `numpy.savez_compressed` 形成的 `npz` 格式文件，包含 3 个 data key，`rays_o` ，`rays_d`，`ranges`，分别代表 lidar 的每个数据点/beam 的光束起点向量、光束方向向量、lidar 在该光束的测量距离。

```python
np.savez_compressed(npz_path, rays_o=rays_o.numpy(), rays_d=rays_d.numpy(), ranges=ranges.numpy())
```

### 坐标系

这里为了简便考虑， `rays_o` 和 `rays_d` 是定义在前面提到的世界坐标系下的。

以 waymo 数据为例，在 lidar 自身坐标系中，其 `rays_o` 就是一组零向量，`rays_d` 则通过 lidar 传感器的 `azimuth`, `inclination` 获得；然后再通过 lidar 的外参和自车姿态，将 `rays_o`，`rays_d` 变换到世界系下。

```python
# Collects raw beam data
# Waymo lidar coordinate system (similar to waymo camera)
rays_o = tf.zeros([*prefix, 3])

cos_azimuth = tf.cos(azimuth)
sin_azimuth = tf.sin(azimuth)
cos_incl = tf.cos(inclination)
sin_incl = tf.sin(inclination)
# NOTE: Waymo lidar coordinate system (similar to waymo camera)
#       facing [+x] direction, z upwards, y left
dx = cos_azimuth * cos_incl
dy = sin_azimuth * cos_incl
dz = sin_incl
rays_d = tf.stack([dx[0],dy[0],dz[0]],axis=-1)

# waymo: [lidar to vehicle]
rays_o = tf.einsum('kr,hwr->hwk', extrinsic[0:3, 0:3], rays_o) + extrinsic[0:3, 3]
rays_d = tf.einsum('kr,hwr->hwk', extrinsic[0:3, 0:3], rays_d)

# waymo: [vehicle to ENU(world)]
if laser.name != dataset_pb2.LaserName.TOP:
    rays_o = tf.einsum('ij,hwj->hwi', frame_pose[:3,:3], rays_o) + frame_pose[:3,3] - world_offset
    rays_d = tf.einsum('ij,hwj->hwi', frame_pose[:3,:3], rays_d)
else:
    rays_o = tf.einsum('hwij,hwj->hwi', frame_pose_per_beam[0,...,:3,:3], rays_o) + frame_pose_per_beam[0,...,:3,3] - world_offset
    rays_d = tf.einsum('hwij,hwj->hwi', frame_pose_per_beam[0,...,:3,:3], rays_d)
```

在上面的例子中，存在对 top lidar 的特殊处理，waymo 做了同样的处理，你们也可以考虑做类似的事情：

因为 lidar 的扫描速度一般比较慢，比如 100 ms 才能完成一次扫描；在这个时间段中，当自车车速稍快时，自车可能已经发生了较大的位移，只使用开始或者结束采集的某一个瞬间的自车姿态来计算所有 lidar 光束是不准的。

对此，waymo 数据对他们的 top lidar ，额外提供了每个lidar 光束 自己的时间戳下的自车的姿态信息。可以推测，这一信息是根据每个lidar光束的时间戳计算的。因此，对于每个 lidar 光束，都可以单独利用该光束对应时刻的自车姿态来计算该光束的世界 `rays_o` 和 `rays_d`。

**注意：这里的世界指的是 ours 偏置后的世界，注意到 `rays_o` 需要减去 `world_offset`**

### 逐帧元数据

因为上面在预处理时，已经存储了世界系下的 lidar 光束坐标，在 `scenario.pt` 中就不需要额外存储什么有意义的 lidar 逐帧信息了。

也就是说，lidar 的 字典中的 `data` 字段实际上是一个空字典。

```python
'observers': { 
    'lidar_0': {
        'id': 'lidar_0', 
        'class_name': 'RaysLidar', 
        'n_frames': 200, 
        'data': {}
    }, 
    ...
}, 
```

这种直接存储世界坐标系下 rays_o 和 rays_d 的做法，事实上杜绝了通过算法 refine lidar 姿态的可能性。因此，其实我们还额外支持单独的 lidar 内参信息、变换信息存储和 refine，这里暂且不表，因为目前大部分代码逻辑和 waymo 是绑定的，需要一定重构。

## 他车/他人/etc. 的格式要求 (如果只有静态背景, 忽略此节)

每一个场景中的动态物体都是 `scenario.pt` 中的 `objects` 字典中的一个 item。

由于其他物体往往只在自车的采集周期中出现若干个不全长的片段，因此 `objects` 首先定义了一个 `segments` 列表 `[]`，然后在列表中放置各个可见片段作为列表的一个个item，每个片段是一个字典 `{}`，包含 `start_frame`，`n_frames` 和 `data` 字段。`data` 字段是个字典，其每个字段存储一个片段帧长的矩阵数据信息。

一个范例：

```python
'objects': { 
    'obj0': {
        'id': 'obj0', 
        'class_name': 'Vehicle', 
        'segments': [ # 片段数据列表，每个item是一个字典。对于某个 他车/他人/etc. 可能会有多个可见片段的数据，不过一般只有一个。
            {
                'start_frame': 12, 
                'n_frames': 30, 
                'data': { # 在这个片段内的逐帧元数据
                    'transform': ..., # [30, 4, 4], np.ndarray, float, 物体到世界坐标系的变换矩阵
                    'scale': ..., # [30, 3], np.ndarray, float, xyz 依次三个方向的尺寸
                }
            }, 
            {
                'start_frame': 100, 
                'n_frames': 40, 
                'data': { # 在这个片段内的逐帧元数据
                    'transform': ..., # [40, 4, 4], np.ndarray, float, 物体到世界坐标系的变换矩阵
                    'scale': ..., # [40, 3], np.ndarray, float, xyz 依次三个方向的尺寸
                }
            }, 
            ... # 更多片段，如果有的话；每个都是一个字典，一个 list 中的 item
        ]
    }, 
    ...
}, 
```

同 camera 和 lidar，这里指的世界是 ours 已经偏置后的世界，需要对 `transform` 的 translation 部分减去 `world_offset`
