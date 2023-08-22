# [neuralsim] Data standards for autonomous driving datasets

English|[中文](autonomous_driving_cn.md)

## Overview

In summary, the final processed data to be loaded for training consists of synchronized storage of images and LiDAR data, with integer frame indices of the same consistent length, along with metadata such as object and ego vehicle poses.

An example of an individual data sequence:

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

The pickle file `scenario.pt` stores all the data except for the raw sensor data, including the ego vehicle's pose, tracklets of other vehicles, and calibration parameters for different sensors.

## `scenario.pt`: Calibrations, annotations and metadata

`scenario.pt` stores a dictionary with four main keys: `observers`, `objects`, `scene_id`, and `metas`.

- `observers` represents all the sensors, primarily cameras and LiDAR. If needed, it can also include information about the ego car.
- `objects` represents all the other objects, excluding the ego car.
- `scene_id` is a string identifier for the scene, mainly used for identification purposes and can be assigned arbitrarily.
- `metas` represents other scene metadata.

Here is an example of this dictionary:

```python
{
    'observers': {  # All the sensors, including cameras and LiDARs
        'camera_0': {
            'id': 'camera_0', 
            'class_name': 'Camera', 
            'n_frames': 200, 
            'data': {...} # The complete frame-wise metadata of the camera. Refer to the camera section for details.
        }, 
        ..., 
        'lidar_0': {
            'id': 'lidar_0', 
            'class_name': 'RaysLidar', 
            'n_frames': 200, 
            'data': {...} # The complete frame-wise metadata of the LiDAR. Refer to the LiDAR section for details.
        }, 
        ..., 
        'ego_car': { # NOTE: Optional
            'id': 'ego_car', 
            'class_name': 'EgoVehicle', 
            'n_frames': 200, 
            'data': {...} # The complete frame-wise metadata of the ego-car.
        }
    }, 
    'objects': { # NOTE: Optional. If this sequence does not have any other vehicles/people/etc., it can be an empty dictionary.
        'obj0': {
            'id': 'obj0', 
            'class_name': 'Vehicle', 
            'segments': # List of segment data, where each segment is a dictionary marked by key `start_frame` and `n_frames`. For a particular vehicle/person/etc., there may be multiple visible segments; but typically only one.
            [
                {
                    'start_frame': 12, 
                    'n_frames': 30, 
                    'data': {...} # Per-frame metadata within this segment
                }, 
                {
                    'start_frame': 100, 
                    'n_frames': 40, 
                    'data': {...} # Per-frame metadata within this segment
                }, 
                ... # More segments, if any; each one is a dictionary and an item in a list.
            ]
        }, 
        'obj1': {...}, 
        ...
    }
    'scene_id': 'your_favorite_seq_name', 
    'metas': {
        'num_frames': 198, # int, total number of frames in the entire scene.
        'world_offset': world_offset, # [3,] np.ndarray, float, the positional difference between the world coordinate system we are using and the original world coordinate system.
        'up_vec': '+z' # Represents the direction of the vertical upward direction in the current coordinate system definition, such as +z representing the positive half-axis direction of the z-axis.
    }
}
```

### World Coordinate System

The pose matrix information of the ego car, all sensors, and all other objects are defined in a **unified world coordinate system**.

- Currently, we have chosen a coordinate system that has the same orientation as the world coordinate system of the original dataset. We define the translation of the ego car at the 0th frame as the origin of the coordinate system.
  - In other words, the translation of the ego car in the 0th frame serves as the offset between the world coordinate system used in our codebase and the original world coordinate system.

  - This offset (a 3D floating-point numpy vector) is stored in `scenario.pt` under `['metas']['world_offset']` for future reference.
- Our codebase supports a coordinate system that can have any orientation.
  - However, in the current training process, we need to know the orientation of the direction opposite to the gravity (i.e., the vertical upward direction) in the coordinate system, such as `+z`, which represents the positive half-axis direction of the z-axis as the vertical upward direction. This information is stored in `scenario.pt` under `['metas']['up_vec']`.

### Frame-wise Metadata

In general, all sensors on the ego car and the ego car itself have metadata for every frame throughout the sequence. On the other hand, observed objects such as other vehicles or pedestrians often have only partial visibility (including visibility under cameras or LiDARs) during certain segments.

Therefore, the storage of frame-wise metadata differs for `observers` and `objects`:

- For `observers`, a `data` dictionary is directly defined within its dictionary, with each field storing matrix data information for the entire duration of the sequence.
  - This is mainly because `observers` are present throughout the entire sequence.

- For `objects`, a `segments` list `[]` is first defined, and each visible segment is placed as an item in the list. Each segment is a dictionary `{}` that includes the `start_frame`, `n_frames`, and `data` fields. The `data` field is a dictionary where each field stores matrix data information for the duration of the segment.
  - This is mainly because other objects often appear in several incomplete segments within the ego car's sampling period.
  - It can be an empty dictionary if the sequence does not have any other vehicles/people/etc.

For specific details on how camera, LiDAR, and other object dictionaries store frame-wise metadata, please refer to the respective sections in the following camera and LiDAR chapters.

### Time Synchronization Requirements

Strict time synchronization is **NOT** required. Only a rough definition of integer frames is needed. Time differences between timestamps of different sensors are allowed. For example, for frame `0000`, the actual acquisition times of `camera_0` and `camera_1` may differ by a few milliseconds.

If it is possible to provide the ego car's pose at the timestamps of different sensor acquisitions, it would be beneficial to include this information. However, if this is not available, the pose refinement process during training can help reduce or eliminate pose errors caused by imprecise timestamps.

We do not require synchronization or correspondence between integer frames of LiDAR and camera data. The supervision signal for LiDAR is used independently in the LiDAR coordinate system, without involving projection into the camera coordinate system. However, currently, it is necessary for the number of frames in both LiDAR and camera data to be the same.

## Cameras

### Image Data

- Images from different cameras can have different sizes.
  - For example, in Waymo Open Dataset, three frontal cameras have `resolution=[1920x1280]`, while two side cameras have `resolution=[1920x886]`. 

- Currently, images at different frames from the same camera need to have the same size.
- Images can be in any mainstream RGB image format. HDR images are not currently supported.

### Frame-wise Metadata

In the dictionary of `scenario.pt`, camera metadata is stored within the dictionary of the `observers` field.

An example of camera metadata:

```python
'observers': { 
    'camera_0': {
        'id': 'camera_0', 
        'class_name': 'Camera', 
        'n_frames': 200, 
        'data': { # Frame-wise metadata of cameras from the beginning to the end
            'hw': ...,  # [200, 2], np.ndarray, long, frame-wise image height and width (in pixels)
            'intr': ..., # [200, 3, 3], np.ndarray, float, intrinsic parameters of the camera for each frame (usually the same for each frame)
            'c2w': ..., # [200, 4, 4], np.ndarray, float, camera to world pose transformation matrix in the OpenCV coordinate system
            'distortion': ..., # [200, ...], np.ndarray, float, optional distortion param (support OpenCV camera, fisheye camera)
        }
    }, 
    ...
}, 
```

#### > Intrinsics

`intr` is a `3x3` pinhole camera intrinsic matrix `[fx, sk, cx; 0, fy, cy; 0, 0, 1]`, where `fx`, `fy`, `cx`, `cy` are in pixel units.

`hw` represents the per-frame image height and width, both in pixel units.

`distortion` is an optional camera distortion parameter. Currently, two definitions are supported: [OpenCV camera](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) and [fisheye camera](https://docs.opencv.org/4.7.0/db/d58/group__calib3d__fisheye.html) (the GPU version of fisheye camera is still under development).

If using [OpenCV camera](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html), the valid size is `[4, 5, 8, 12, 14]`, with the following symbols: $(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])$

If using [fisheye camera](https://docs.opencv.org/4.7.0/db/d58/group__calib3d__fisheye.html), the valid size is `[4]`, with the following symbols: $(k_1,k_2,k_3,k_4)$

#### > Camera pose: `c2w`

The `c2w` matrix defines the pose transformation from the camera coordinate system (in the OpenCV convention) to the world coordinate system (with a unified world offset applied).

If the camera is not defined in the OpenCV coordinate system, you will need to perform a transformation using the matrix that converts from the OpenCV coordinate system to your coordinate system.

```python
c2w = c2w_of_yours @ opencv_to_yours
```

Here, we provide an example converting a Waymo `c2w` to OpenCV `c2w`: 

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

In this context, `v2w` represents the vehicle pose defined by Waymo data, which is the transformation matrix from the vehicle coordinate system to the world coordinate system. `c2v` refers to the the inverse of camera extrinsics annotated in Waymo data, representing the camera pose in the vehicle coordinate system (or the transformation matrix from the camera to the vehicle).

In the above script, to obtain `c2w` matrix, we first use `opencv_to_waymo` to transform vectors in the OpenCV camera coordinate system to the Waymo camera coordinate system. Then, we further left-multiply `v2w @ c2v` to transform vectors in Waymo camera to the Waymo world coordinate system. Subsequently, the resulting `c2w` matrix can transform vectors in OpenCV cameras to Waymo world.

>  NOTE: Here, "world" refers to our world with offsets from the original world. `v2w` needs to be subtracted by `world_offset`.

## LiDARs

We need access to the original LiDAR beam direction vectors and range measurements, because we view LiDARs as sparse depth sensors (instead of pointcloud data). This approach is particularly useful for surface reconstruction that involves multiple dynamic objects.

### LiDAR Data

Specifically, each frame of lidar data will be stored in an `.npz` file created using `numpy.savez_compressed`. The file will contain three data keys: `rays_o`, `rays_d`, and `ranges`. These keys represent the origin vectors of each LiDAR data point/beam (`rays_o`), the direction vectors of the LiDAR beams (`rays_d`), and the measured distances of the lidar for each beam (`ranges`).

```python
# To save
np.savez_compressed(npz_path, rays_o=rays_o.numpy(), rays_d=rays_d.numpy(), ranges=ranges.numpy())

# To load
data_dict = np.load(npz_path)
rays_o = data_dict['rays_o']
rays_d = data_dict['rays_d']
ranges = data_dict['ranges']
```

### Coordinate System

For simplicity, `rays_o` and `rays_d` are defined in the world coordinate system defined earlier.

#### > Taking Waymo Open Dataset as an example

In LiDARs' own coordinate systems, `rays_o` is a set of zero vectors, while `rays_d` is obtained using the lidar sensor's `azimuth` and `inclination`. 

Then, `rays_o` and `rays_d` are transformed to the world coordinate system using the lidar's extrinsics and the vehicle's pose.

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

In the example above, there is a special treatment for the top LiDAR, which is implemented by Waymo, and readers can also consider doing something similar to improve accuracy:

Since the lidar scanning speed is generally slow, for example, it takes around 100 ms to complete one scan. During this time period, if the vehicle speed is relatively fast, the vehicle may have undergone significant displacement. It is inaccurate to calculate all lidar beams using only the vehicle pose at the beginning or end of the data acquisition.

To address this, Waymo provides additional information for each LiDAR beam of their top LiDAR, which includes the vehicle pose at the timestamp corresponding to each LiDAR beam. This information may be calculated and interpolated using the timestamp of each LiDAR beam. 

Subsequently, each LiDAR beam's `rays_o` and `rays_d` in world can be individually calculated.

>  NOTE: Here, "world" refers to our world with offsets from the original world. `rays_o` needs to be subtracted by `world_offset`.

### Frame-wise Metadata

Since the world coordinate system of the LiDAR beams has already been stored during preprocessing as above mentioned, there is no meaningful frame-wise LiDAR information in `scenario.pt`. That is to say, the `data` field in the LiDAR dictionary is actually an empty dictionary.

```python
'observers': { 
    'lidar_0': {
        'id': 'lidar_0', 
        'class_name': 'RaysLidar', 
        'n_frames': 200, 
        'data': {} # Empty dict
    }, 
    ...
}, 
```

> NOTE: The approach of directly storing `rays_o` and `rays_d` in the world coordinate system actually eliminates the possibility of refining the lidar pose through algorithms. However, we do provide additional support for storing and refining individual lidar intrinsic parameters and transformation information separately. This will not be discussed here as the majority of the code logic is currently tied to Waymo and would require significant refactoring. You can read our codes on waymo dataset [preprocess.py](../../dataio/autonomous_driving/waymo/preprocess.py) for more details.

## Other Vehicles/People/etc.

> NOTE: If only dealing with static backgrounds, ignore this section

Each dynamic object in a scene is an item in the `objects` dictionary in `scenario.pt`.

Since other objects often only appear in several incomplete segments within the vehicle's data collection cycle, the `objects` dictionary first defines a `segments` list `[]`. Each visible segment is then placed as an item in the list, where each segment is a dictionary `{}` that includes `start_frame`, `n_frames`, and `data` fields. The `data` field is a dictionary where each field stores matrix data information for a segment frame length.

Here is an example:

```python
'objects': { 
    'obj0': {
        'id': 'obj0', 
        'class_name': 'Vehicle', 
        'segments': [ # List of segment data, each segment is a dictionary. There may be multiple visible segments for a certain vehicle/person/etc., but usually only one.
            {
                'start_frame': 12, 
                'n_frames': 30, 
                'data': { # Frame-wise metadata within this segment
                    'transform': ..., # [30, 4, 4] np.ndarray, float, transformation matrix from object to world coordinate system
                    'scale': ..., # [30, 3] np.ndarray, float, sizes in the xyz directions respectively
                }
            }, 
            {
                'start_frame': 100, 
                'n_frames': 40, 
                'data': { # Frame-wise metadata within this segment
                    'transform': ..., # [40, 4, 4], np.ndarray, float, transformation matrix from object to world coordinate system
                    'scale': ..., # [40, 3], np.ndarray, float, sizes in the xyz directions respectively
                }
            }, 
            ... # More segments, if any; each is a dictionary & an item in the list
        ]
    }, 
    ...
}, 
```

>  NOTE: Similar to camera and LiDAR, when referring to the "world" here, it means the world that has been adjusted and offset by "world_offset". Therefore, for the translation part of the transform, it needs to be subtracted by "world_offset".