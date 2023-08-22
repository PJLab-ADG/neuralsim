# [neuralsim] BlendedMVS dataset

## Usage

### Download

#### > Tested small split (187MB) (already normalized)

[Google Drive](https://drive.google.com/file/d/13hpXTYtjXNDu1HyFZSy66-ZF1H6wFPYW/view?usp=drive_link)

#### > Full (27.5GB)

Download the [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset via [this link](https://github.com/YoYo000/BlendedMVS#download).

For now, we have tested on the `low-res set` of the `BlendedMVS` split (27.5GB).

### Normalization

> This is only for the original BMVS dataset. We have provided a small tested split with already normalized cams. See links above.

The camera poses in [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset have random centers and random scales. Therefore, it is necessary to first normalize the camera centers to the origin and standardize the scale of the camera distances.

In order to achieve this, we have developed a simple script to find the focus center of **multiple object-centric views**, by minimizing the average projected pixel distance w.r.t. the image centers across all frames. This script only requires camera intrinsics and poses.

```shell
cd /path/to/neuralsim
python dataio/bmvs/normalize_bmvs.py --root /path/to/bmvs
```

It will generate `cameras_sphere.npz` files in the instance directories:

```
/path/to/bmvs
├── 5aa0f9d7a9efce63548c69a1
│   ├── cameras_sphere.npz
│   ├── blended_image
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ├── ...
│   ├── cams
│   │   ├── 00000000_cam.txt
│   │   ├── 00000001_cam.txt
│   │   ├── ...
├── 5aa235f64a17b335eeaf9609
│   ├── cameras_sphere.npz
│   ├── ...
├── ...
```

## DEBUG

```shell
cd /path/to/neuralsim
source set_env.sh
python dataio/bmvs/bmvs_dataset.py
```
