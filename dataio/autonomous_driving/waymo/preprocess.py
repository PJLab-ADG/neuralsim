"""
@file   preprocess.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Waymo dataset preprocess.
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
set_env(3)

import io
import os
import pickle
import functools
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List

from dataio.autonomous_driving.waymo.filter_dynamic import stat_dynamic_objects
from dataio.autonomous_driving.waymo.waymo_dataset import *

def process_single_sequence(
    sequence_file: str, 
    out_root: str, 
    rgb_dirname: str = None, 
    lidar_dirname: str = None, 
    pcl_dirname: str = None,
    mask_dirname: str = None, 
    # Other configs
    class_names: List[str] = WAYMO_CLASSES, 
    should_offset_pos=True,
    should_offset_timestamp=True, 
    should_process_gt=True,
    ignore_existing=False, 
    ):
    
    # NOTE: 
    # 1. It seems that tensorflow==2.11 is no longer thread safe (compared to tf==2.6.0); 
    #   Using multi-threading causes tons of errors randomly everywhere !!! TAT
    # 2. Hence, we need to use multi-processing instead of multi-threading;
    #   In this case, we need to import tensorflow (and any module that will import tensorflow inside) 
    #       in process function instead of globally, to prevent CUDA initialization BUG.
    #   Multi-processing consumes more GPU mem even with set_memory_growth, compared to multi-threading.
    
    # NOTE: For tensorflow>=2.2 (2.11.0 currently)
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
    from dataio.autonomous_driving.waymo.waymo_filereader import WaymoDataFileReader
    
    if not os.path.exists(sequence_file):
        print(f"Not exist: {sequence_file}")
        return
    
    try:
        
        # dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
        dataset = WaymoDataFileReader(str(sequence_file))

        scene_objects = dict()
        scene_observers = dict()

        #---- Use frame0 to process some meta info
        # frame0_data = bytearray(next(iter(dataset)).numpy())
        # frame0 = dataset_pb2.Frame()
        # frame0.ParseFromString(frame0_data)
        frame0 = next(iter(dataset))
        
        # scene_id = frame0.context.name
        scene_id = file_to_scene_id(sequence_file)

        #---- Outputs
        os.makedirs(os.path.join(out_root, scene_id), exist_ok=True)
        
        rgb_dir = os.path.join(out_root, scene_id, rgb_dirname) if rgb_dirname else None
        lidar_dir = os.path.join(out_root, scene_id, lidar_dirname) if lidar_dirname else None
        pcl_dir = os.path.join(out_root, scene_id, pcl_dirname) if pcl_dirname else None
        scenario_fpath = os.path.join(out_root, scene_id, "scenario.pt")
        if ignore_existing:
            if (rgb_dir is None) or os.path.exists(rgb_dir): rgb_dir = None
            if (lidar_dir is None) or os.path.exists(lidar_dir): lidar_dir = None
            if (pcl_dir is None) or os.path.exists(pcl_dir): pcl_dir = None
        
        # NOTE: To normalize segments poses (for letting x=0,y=0,z=0 @ 0-th frame)
        world_offset = np.zeros([3,])
        if should_offset_pos:
            #---- OPTION1: Use the camera_0's 0-th pose as offset
            # extr00 = np.array(frame0.context.camera_calibrations[0].extrinsic.transform).reshape(4,4)
            # pose00 = np.array(frame0.images[0].pose.transform).reshape(4,4)
            # c2w00 = pose00 @ extr00
            # world_offset = c2w00[:3, 3]
            
            #---- OPTION2: Use the vehicle's 0-th pose as offset (for waymo, the same with OPTION1: waymo's frame.pose is exactly camera0's pose)
            frame0_pose = np.array(frame0.pose.transform, copy=True).reshape(4,4)
            world_offset = frame0_pose[:3, 3]
        timestamp_offset = 0
        if should_offset_timestamp:
            timestamp_offset = frame0.timestamp_micros / 1e6

        frame_timestamps = []

        #------------------------------------------------------
        #--------    Dynamic object statistics     ------------
        #------------------------------------------------------
        dynamic_stats = stat_dynamic_objects(dataset)

        # NOTE: Not used.
        # frame_inds_with_panoptic_label = []
        # for frame_ind, frame in enumerate(dataset):
        #     if frame.images[0].camera_segmentation_label.panoptic_label:
        #         frame_inds_with_panoptic_label.append(frame_ind)

        #--------------- per-frame processing
        # for frame_ind, framd_data in enumerate(dataset):
            # frame = dataset_pb2.Frame()
            # frame.ParseFromString(bytearray(framd_data.numpy()))
        for frame_ind, frame in enumerate(tqdm(dataset, f"processing...")):
            #---- Ego pose
            frame_pose = np.array(frame.pose.transform, copy=True).reshape(4,4)
            frame_pose[:3, 3] -= world_offset
            frame_timestamp = frame.timestamp_micros / 1e6
            if should_offset_timestamp:
                frame_timestamp -= timestamp_offset
            frame_timestamps.append(frame_timestamp)

            #------------------------------------------------------
            #--------------     Frame Observers      --------------
            #------------------------------------------------------
            if 'ego_car' not in scene_observers:
                scene_observers['ego_car'] = dict(
                    class_name='EgoVehicle', n_frames=0, 
                    data=dict(v2w=[], global_timestamps=[], global_frame_inds=[]))
            scene_observers['ego_car']['n_frames'] += 1
            scene_observers['ego_car']['data']['v2w'].append(frame_pose)
            scene_observers['ego_car']['data']['global_timestamps'].append(frame_timestamp)
            scene_observers['ego_car']['data']['global_frame_inds'].append(frame_ind)

            #------------------------------------------------------
            #------------------     Cameras      ------------------
            # NOTE: !!! Waymo's images order is not 12345 !!!
            # frame.context.camera_calibrations[0,1,2,3,4].name:[1,2,3,4,5]
            # frame.images[0,1,2,3,4].name:                     [1,2,4,3,5]
            for j in range(len(WAYMO_CAMERAS)):
                c = frame.context.camera_calibrations[j]
                for _j in range(len(frame.images)):
                    if frame.images[_j].name == c.name:
                        break
                camera = frame.images[_j]
                assert c.name == camera.name == (j+1)
                str_id = idx_to_camera_id(_j)
                
                camera_timestamp = camera.pose_timestamp
                if should_offset_timestamp:
                    camera_timestamp -= timestamp_offset
                
                h, w = c.height, c.width
                
                # fx, fy, cx, cy, k1, k2, p1, p2, k3
                fx, fy, cx, cy, *distortion = np.array(c.intrinsic)
                distortion = np.array(distortion)
                intr = np.eye(3)
                intr[0,0] = fx
                intr[1,1] = fy
                intr[0,2] = cx
                intr[1,2] = cy
                
                """
                    < opencv / colmap convention >                 --->>>   < waymo convention >
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
                # NOTE: Opencv camera to waymo camera
                opencv_to_waymo = np.eye(4)
                opencv_to_waymo[:3 ,:3] = np.array(
                    [[0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0]])
                
                # NOTE: Waymo: extrinsic=[camera to vehicle]
                c2v = np.array(c.extrinsic.transform).reshape(4,4)
                # NOTE: Waymo: pose=[vehicle to ENU(world)]
                v2w = np.array(camera.pose.transform).reshape(4,4)
                v2w[:3, 3] -= world_offset
                # NOTE: [camera to ENU(world)]
                c2w = v2w @ c2v @ opencv_to_waymo
                
                if str_id not in scene_observers:
                    scene_observers[str_id] = dict(
                        class_name='Camera', n_frames=0, 
                        data=dict(hw=[], intr=[], distortion=[], c2v_0=[], c2v=[], sensor_v2w=[], c2w=[], 
                                  global_timestamps=[], global_frame_inds=[]))
                scene_observers[str_id]['n_frames'] += 1
                scene_observers[str_id]['data']['hw'].append((h,w))
                scene_observers[str_id]['data']['intr'].append(intr)
                scene_observers[str_id]['data']['distortion'].append(distortion)
                scene_observers[str_id]['data']['c2v_0'].append(c2v)
                scene_observers[str_id]['data']['c2v'].append(c2v @ opencv_to_waymo)
                scene_observers[str_id]['data']['sensor_v2w'].append(v2w) # v2w at each camera's timestamp
                scene_observers[str_id]['data']['c2w'].append(c2w)
                scene_observers[str_id]['data']['global_timestamps'].append(camera_timestamp)
                scene_observers[str_id]['data']['global_frame_inds'].append(frame_ind)

                #-------- Process observation groundtruths
                if should_process_gt and rgb_dir:
                    img = Image.open(io.BytesIO(camera.image))
                    assert [*(np.asarray(img)).shape[:2]] == [h, w]
                    img_cam_dir = os.path.join(rgb_dir, str_id)
                    os.makedirs(img_cam_dir, exist_ok=True)
                    img.save(os.path.join(img_cam_dir, idx_to_img_filename(frame_ind)))

            #------------------------------------------------------
            #------------------     Lidars      -------------------
            if should_process_gt and pcl_dir:
                points = []
                points_intensity = []
                points_elongation = []
                points_NLZ = []
            
            # frame.context.laser_calibrations[0,1,2,3,4].name: [2,5,3,4,1]
            # frame.lasers[0,1,2,3,4].name:                     [1,2,3,4,5]
            laser_calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
            ri_index = 0
            for j in range(len(WAYMO_LIDARS)):
                c = laser_calibrations[j]            
                laser = frame.lasers[j]
                assert c.name == laser.name == (j+1)
                str_id = idx_to_lidar_id(j)
                
                # NOTE: Waymo assumes LiDARs are all captured at frame timestamp.
                #       The rolling shutter effect of TOP LiDAR is compensated by \
                #           the per-beam ego pose `laser_return.range_image_pose_compressed`, 
                #           which is processed below.
                lidar_timestamp = frame_timestamp
                
                # Waymo: extrinsic=[lidar to vehicle]
                extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])
                l2w = frame_pose @ extrinsic
                
                if str_id not in scene_observers:
                    scene_observers[str_id] = dict(
                        class_name='RaysLidar', n_frames=0, 
                        data=dict(l2v=[], l2w=[], global_timestamps=[], global_frame_inds=[]))
                scene_observers[str_id]['n_frames'] += 1
                scene_observers[str_id]['data']['l2v'].append(extrinsic)
                scene_observers[str_id]['data']['l2w'].append(l2w)
                # scene_observers[str_id]['data']['sensor_v2w'].append(frame_pose)
                scene_observers[str_id]['data']['global_timestamps'].append(lidar_timestamp)
                scene_observers[str_id]['data']['global_frame_inds'].append(frame_ind)
                
                if should_process_gt and (lidar_dir or pcl_dir):
                    if ri_index == 0:
                        laser_return = laser.ri_return1
                    elif ri_index == 1:
                        laser_return = laser.ri_return2
                    else:
                        raise ValueError(f"Invalid ri_index={ri_index}")
                    assert len(laser_return.range_image_compressed) > 0
                    range_image_str_tensor = tf.io.decode_compressed(laser_return.range_image_compressed, 'ZLIB')
                    range_image = dataset_pb2.MatrixFloat()
                    range_image.ParseFromString(bytearray(range_image_str_tensor.numpy()))
                    range_image_tensor = tf.reshape(tf.convert_to_tensor(range_image.data), range_image.shape.dims)
                    
                    # H, W = range_image_tensor.shape[:2]
                    prefix = range_image_tensor.shape[:2]
                    
                    if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                        beam_inclinations = range_image_utils.compute_inclination(
                            tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                            height=range_image.shape.dims[0])
                    else:
                        beam_inclinations = tf.constant(c.beam_inclinations)
                    
                    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
                    
                    # [1, H, W, 3]
                    range_image_polar = range_image_utils.compute_range_image_polar(
                        range_image=tf.expand_dims(range_image_tensor[..., 0], axis=0),
                        extrinsic=tf.expand_dims(extrinsic, axis=0),
                        inclination=tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0))
                    # [1, H, W]
                    azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)
                    
                    if laser.name == dataset_pb2.LaserName.TOP:
                        range_image_top_pose_str_tensor = tf.io.decode_compressed(laser_return.range_image_pose_compressed, 'ZLIB')
                        range_image_top_pose = dataset_pb2.MatrixFloat()
                        range_image_top_pose.ParseFromString(bytearray(range_image_top_pose_str_tensor.numpy()))
                        # [H, W, 6]
                        range_image_top_pose_tensor = tf.reshape(tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims)
                        # [H, W, 3, 3]
                        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
                            range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
                            range_image_top_pose_tensor[..., 2])
                        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
                        # [H, W, 4, 4]
                        pixel_pose_local = transform_utils.get_transform(
                            range_image_top_pose_tensor_rotation,
                            range_image_top_pose_tensor_translation)
                        pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                        frame_pose_local = tf.expand_dims(tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4])), axis=0)
                    else:
                        pixel_pose_local = None
                        frame_pose_local = None
                    
                    range_image_mask = range_image_tensor[..., 0] > 0
                    range_image_intensity = range_image_tensor[..., 1]
                    range_image_elongation = range_image_tensor[..., 2]
                    range_image_NLZ = range_image_tensor[..., 3]
                    
                    if should_process_gt and lidar_dir:
                        #---- Collects raw beam data
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

                        #---- Save rays_o, rays_d, and raw laser beam data
                        lidar_cur_dir = os.path.join(lidar_dir, str_id)
                        os.makedirs(lidar_cur_dir, exist_ok=True)
                        lidar_cur_fpath = os.path.join(lidar_cur_dir, idx_to_lidar_filename(frame_ind))
                        if pixel_pose_local is not None:
                            # #---- Optionally downsample on scans (waymo TOP: 64x2650; others 200x600)
                            # ds_vertical = 4
                            # ds_horizonal = 1
                            # rays_o = rays_o[::ds_vertical, ::ds_horizonal]
                            # rays_d = rays_d[::ds_vertical, ::ds_horizonal]
                            # pixel_pose_local = pixel_pose_local[:, ::ds_vertical, ::ds_horizonal]
                            # range_image_range = range_image_range[:, ::ds_vertical, ::ds_horizonal]
                            # range_image_top_pose_tensor = range_image_top_pose_tensor[::ds_vertical, ::ds_horizonal]
                            
                            # Waymo: _pixel_pose_local=[vehicle to ENU(world)]
                            mask_valid = tf.reduce_all(range_image_top_pose_tensor!=0, axis=-1).numpy()
                            rays_o = rays_o[mask_valid][None,...]
                            rays_d = rays_d[mask_valid][None,...]
                            _pixel_pose_local = pixel_pose_local[0].numpy()[mask_valid][None,...]
                            _range_image_range = range_image_range[0].numpy()[mask_valid][None,...]
                            _pixel_pose_local[...,:3,3] -= world_offset
                            
                            # NOTE: Delta-pose the ray to account for ego-car motion during delta timestamps
                            dpose = np.linalg.inv(frame_pose @ extrinsic) @ _pixel_pose_local @ extrinsic
                            
                            #-------- OPTION1: save original rays & dpose
                            # np.savez_compressed(lidar_cur_fpath, rays_o=rays_o, rays_d=rays_d, ranges=_range_image_range, dpose=dpose)

                            #-------- OPTOIN2: directly saved modified rays; also save dpose just in case of need.
                            rays_o = tf.einsum('hwij,hwj->hwi', dpose[...,:3,:3], rays_o) + dpose[...,:3,3]
                            rays_d = tf.einsum('hwij,hwj->hwi', dpose[...,:3,:3], rays_d)
                            np.savez_compressed(lidar_cur_fpath, rays_o=rays_o.numpy().astype(np.float32), rays_d=rays_d.numpy().astype(np.float32), ranges=_range_image_range.astype(np.float32), dpose=dpose.astype(np.float32))
                        else:
                            _range_image_range = range_image_range[0].numpy()
                            np.savez_compressed(lidar_cur_fpath, rays_o=rays_o.numpy().astype(np.float32), rays_d=rays_d.numpy().astype(np.float32), ranges=_range_image_range.astype(np.float32))
                
                    if should_process_gt and pcl_dir:
                        # TODO:
                        # 1. Save pcl in the right place.
                        # 2. Add option to ignore object points (background only)
                        
                        # Collect point clouds data
                        # [1, H, W, 3]
                        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                            tf.expand_dims(range_image_tensor[..., 0], axis=0),
                            tf.expand_dims(extrinsic, axis=0),
                            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                            pixel_pose=pixel_pose_local,
                            frame_pose=frame_pose_local)
                        
                        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
                        points_tensor = tf.gather_nd(range_image_cartesian, tf.where(range_image_mask)) - world_offset
                        points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
                        points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
                        points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
                        
                        points.append(points_tensor.numpy())
                        points_intensity.append(points_intensity_tensor.numpy())
                        points_elongation.append(points_elongation_tensor.numpy())
                        points_NLZ.append(points_NLZ_tensor.numpy())

            #------------------------------------------------------
            #---------------     Frame Objects      ---------------
            #------------------------------------------------------
            for l in frame.laser_labels:
                str_id = str(l.id)
                # str_id = f"{scene_id}#{l.id}"
                
                if WAYMO_CLASSES[l.type] not in class_names:
                    continue
                
                if str_id not in scene_objects:
                    scene_objects[str_id] = dict(
                        id=l.id,
                        # class_ind=l.type,
                        class_name=WAYMO_CLASSES[l.type],
                        frame_annotations=[]
                    )
                
                # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
                box = l.box
                
                # Box coordinates in vehicle frame.
                tx, ty, tz = box.center_x, box.center_y, box.center_z
                
                # The heading of the bounding box (in radians).  The heading is the angle
                #   required to rotate +x to the surface normal of the box front face. It is
                #   normalized to [-pi, pi).
                c = np.math.cos(box.heading)
                s = np.math.sin(box.heading)
                
                # [object to vehicle]
                # https://github.com/gdlg/simple-waymo-open-dataset-reader/blob/d488196b3ded6574c32fad391467863b948dfd8e/simple_waymo_open_dataset_reader/utils.py#L32
                o2v = np.array([
                    [ c, -s,  0, tx],
                    [ s,  c,  0, ty],
                    [ 0,  0,  1, tz],
                    [ 0,  0,  0,  1]])
                
                # [object to ENU world]
                pose = frame_pose @ o2v
                
                # difficulty = l.detection_difficulty_level
                
                # tracking_difficulty = l.tracking_difficulty_level
                
                # Dimensions of the box. length: dim x. width: dim y. height: dim z.
                # length: dim_x: along heading; dim_y: verticle to heading; dim_z: verticle up
                dimension = [box.length, box.width, box.height]
                
                scene_objects[str_id]['frame_annotations'].append(
                    [[frame_ind, frame_timestamp], [pose, dimension]]
                )

        n_global_frames = frame_ind + 1

        #--------------- Per-observer processing
        for oid, odict in scene_observers.items():
            for k, v in odict['data'].items():
                odict['data'][k] = np.array(v)

        #--------------- Per-object processing: from frame annotations to frame attribute segments
        for oid, odict in scene_objects.items():
            obj_annos = odict.pop('frame_annotations')
            
            segments = []
            for i, ([frame_ind, frame_timestamp], [pose, dimension]) in enumerate(obj_annos):
                if (i == 0) or (frame_ind - obj_annos[i-1][0][0] != 1):
                    cur_segment = dict(
                        start_frame=frame_ind,
                        n_frames=None,
                        data=None,
                    )
                    cur_seg_data = dict(
                        transform=[],
                        scale=[],
                        global_timestamps=[],
                        global_frame_inds=[]
                    )

                # NOTE: Waymo assumes all annotations are captured at frame timestamp.
                cur_seg_data['global_timestamps'].append(frame_timestamp)
                
                cur_seg_data['transform'].append(pose)
                cur_seg_data['scale'].append(dimension)
                cur_seg_data['global_frame_inds'].append(frame_ind)
                
                if (i == len(obj_annos)-1) or (obj_annos[i+1][0][0] - frame_ind != 1):
                    #----------------- Process last segment
                    for k, v in cur_seg_data.items():
                        cur_seg_data[k] = np.array(v)
                    cur_segment['n_frames'] = frame_ind - cur_segment['start_frame'] + 1
                    cur_segment['data'] = cur_seg_data
                    segments.append(cur_segment)
            
            odict['n_full_frames'] = n_global_frames
            odict['segments'] = segments

        scenario = dict()
        scenario['scene_id'] = scene_id
        scenario['metas'] = {
            'n_frames': n_global_frames, 
            'world_offset': world_offset, 
            'timestamp_offset': timestamp_offset, 
            'frame_timestamps': np.array(frame_timestamps), 
            'dynamic_stats': dynamic_stats, 
        }
        scenario['objects'] = scene_objects
        scenario['observers'] = scene_observers
        
        with open(scenario_fpath, 'wb') as f:
            pickle.dump(scenario, f)
            print(f"=> scenario saved to {scenario_fpath}")

    except Exception as e:
        print(f"Process waymo run into error: \n{e}")
        raise e
    
    return True

def create_dataset(
    root: str, 
    seq_list_fpath: str, 
    out_root: str, 
    *, 
    j: int=8, 
    should_offset_pos=True, 
    should_process_gt=True, 
    ignore_existing=False):
    import concurrent.futures as futures
    from tqdm.contrib.concurrent import process_map, thread_map
    
    os.makedirs(out_root, exist_ok=True)
    
    seq_fpath_list = parse_seq_file_list(root, seq_list_fpath=seq_list_fpath)
    num_workers = min(j, len(seq_fpath_list))
    process_fn = functools.partial(
        process_single_sequence, 
        out_root=out_root, 
        rgb_dirname="images", 
        lidar_dirname="lidars", 
        pcl_dirname=None, 
        should_offset_pos=should_offset_pos, 
        should_process_gt=should_process_gt, 
        ignore_existing=ignore_existing
    )
    
    if num_workers == 1:
        for seq_fpath in tqdm(seq_fpath_list, 'Processing waymo...'):
            process_fn(seq_fpath)
    else:
        process_map(process_fn, seq_fpath_list, max_workers=args.j, desc='Processing waymo...')
        
        # with futures.ThreadPoolExecutor(num_workers) as executor:
        #     iterator = executor.map(process_fn, seq_fpath_list)
        #     next(iterator)

if __name__ == "__main__":
    """
    Usage:
        python preprocess.py --root /path/to/waymo/training --out_root /path/to/processed --seq_list /path/to/xxx.lst -j8
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="/media/guojianfei/DataBank0/dataset/waymo/training", required=True, 
        help="Root directory of raw .tfrecords")
    parser.add_argument(
        "--seq_list", type=str, default=None, 
        help="Optional specify subset of sequences. If None, will process all sequences contained in args.root")
    parser.add_argument(
        "--out_root", type=str, default="/data1/waymo/processed", required=True, 
        help="Output root directory")
    parser.add_argument("--no_offset_pose", action="store_true")
    parser.add_argument("--no_process_gt", action="store_true")
    parser.add_argument("--ignore_existing", action="store_true")
    parser.add_argument('-j', type=int, default=4, help='max num workers')
    args = parser.parse_args()
    create_dataset(args.root, args.seq_list, args.out_root, j=args.j, should_offset_pos=not args.no_offset_pose, should_process_gt=not args.no_process_gt, ignore_existing=args.ignore_existing)