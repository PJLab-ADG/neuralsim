
# [neuralsim] Waymo Open Dataset

https://github.com/PJLab-ADG/neuralsim/assets/25529198/42080c6e-6bde-4ecf-8d42-55b5fddf7be9

## v1.x API of perception training split

### Download

We use Waymo Open Dataset Perception - Perception - training split's v1.x versions. If you have already downloaded older versions, skip this step since older versions of v1.x are also suitable.

Go to https://waymo.com/open/ .

Download Perception - v1.4.2 - training split by following their instructions.

Before proceeding, please make sure to:

1. Fill out the Waymo Terms of Use agreement
2. Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install) 
3. Complete `gcloud auth login`

:pushpin: For [StreetSurf](https://ventusff.github.io/streetsurf_web/) paper, you can choose to only download the selected 32 sequences by:

```shell
cd /path/to/neuralsim/dataio/autonomous_driving/waymo
bash download_waymo.sh waymo_static_32.lst /path/to/waymo/training
```

### Preprocess

The original data of the [Waymo Open Dataset - Perception](https://waymo.com/open/data/perception/) - training splits consists of raw `.tfrecord` files that encode sensor data, annotations, and calibrations.

In order to efficiently and conveniently load this data for training, we preprocess these files into separate formats. These formats include camera images, LiDAR `.npz` data, and scenario pickle `.pt` files that contain converted calibration results and tracklet transformations.

In fact, we have standardized the preprocessed datasets into **universal formats applicable to all autonomous driving datasets**. For more details on this universal format, please refer to [docs/data/autonomous_driving.md](../../../docs/data/autonomous_driving.md)

#### > Install requirements to preprocess the waymo dataset

```shell
pip install tensorflow_gpu==2.11.0 waymo-open-dataset-tf-2-11-0
```

NOTE:

- Processing waymo data is the last and only place where tensorflow is required in neuralsim. Feel free to uninstall it afterwards.

- You can check or use [env_backup.yml](env_backup.yml), a backup of our environment, if you run into any troubles.

  - First modify `name` and `prefix` according to your situation. Then run `conda env create -f env_backup.yml`.
  - Usually this is not needed if you follow our steps when installing [nr3d_lib](https://github.com/PJLab-ADG/nr3d_lib). 

- Failsafe: if all above fails, try to downgrade to tensorflow==2.6.0 which we find less buggy.

  - :warning:  This will break numpy dependencies by installing numpy==1.19. You can create a new conda env, or uninstall tensorflow and re-install back numpy>1.20 after preprocess is done.
  
  - ```shell
    pip install tensorflow_gpu==2.6.0 waymo-open-dataset-tf-2-6-0 protobuf==3.20
    ```
  

#### > Run the preprocess script

You can run the script in parallel by specifying `-j4` argument, where `4` indicates the number of processes in parallel.

:pushpin: For [StreetSurf](https://ventusff.github.io/streetsurf_web/) paper: additionally specify `--seq_list=waymo_static_32.lst`

 ```shell
 cd /path/to/neuralsim/dataio/autonomous_driving/waymo
 python preprocess.py --root=/path/to/waymo/training --out_root=/path/to/waymo/processed -j4 --seq_list=waymo_static_32.lst
 ```

NOTE:

- If your data is stored on portable hard drives, specifying too many parallel processes may cause the program to hang forever.
- You can always specify `-j1` for sequential running instead of parallel if you run into any problems with parallel execution or want to debug.

### Extract monocular normals & depths priors

As stated in [StreetSurf](https://ventusff.github.io/streetsurf_web/) section 3.3.2, when LiDAR data is not available, we use monocular normal and depth cues inferred by [omnidata](https://github.com/EPFL-VILAB/omnidata).

NOTE:

- Normal cues are generally more important than depth cues. In most cases, using only normal cues is sufficient.
- The scale and shift of monocular depths have no correlation with real-world depths. They can only be indirectly used as weak hints.
- The inferred normals are in range `[-1,1]`. The inferred depths are typically in range `[0,1]`.

#### > Setup `omnidata`

Clone `omnidata` and install requirements.

```shell
# Clone omnidata into your favorite directory
git clone https://github.com/EPFL-VILAB/omnidata

# Install dependencies
pip install einops joblib pandas h5py scipy seaborn kornia timm pytorch-lightning
```

Download pretrained models following [download weights and code](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch#pretrained-models).

NOTE: 

- If you encounter `gdown` error - access denied, try this answer: https://github.com/EPFL-VILAB/omnidata/issues/52#issuecomment-1627969898

#### > Extract mono cues

:pushpin: For [StreetSurf](https://ventusff.github.io/streetsurf_web/) paper: additionally specify `--seq_list=waymo_static_32.lst`

```shell
cd /path/to/neuralsim/dataio/autonomous_driving/waymo

# Extract depth
python extract_mono_cues.py --task=depth --data_root=/path/to/waymo/processed --omnidata_path=/path/to/omnidata/omnidata_tools/torch/ --seq_list=waymo_static_32.lst

# Extract normals
python extract_mono_cues.py --task=normal --data_root=/path/to/waymo/processed --omnidata_path=/path/to/omnidata/omnidata_tools/torch/ --seq_list=waymo_static_32.lst
```

NOTE: You can pass `--verbose` and `--ignore_existing` if needed.

 ### Extract mask priors -  for sky, pedestrian etc.

As stated in [StreetSurf](https://ventusff.github.io/streetsurf_web/) section 3.3.1, it is recommended to utilize sky masks to further distinguish sky models from the distant-view model.

Although Waymo provides panoptic annotation in their data, it does not cover full frames of each sequence (in fact, only about 10+ frames out of 200 have this annotation) and is insufficient.

Hence, we employ [SegFormer](https://github.com/NVlabs/SegFormer) to effectively infer semantic segmentation masks and extract sky masks from them.

NOTE:

- We use the `cityscapes` taxonomy.
- Semantic segmentation is sufficient. Instance segmentation is not required.
- Though there are models with better metrics, what is needed here is an efficient solution and a better and more stable performance on sky segmentation.

#### > Setup a seperate conda env for `SegFormer`

:warning: SegFormer relies on `mmcv-full=1.2.7`, which relies on `pytorch=1.8` (pytorch<1.9). Hence, a seperate conda env is required.

```shell
#-- Set conda env
conda create -n segformer python=3.8
conda activate segformer
# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#-- Install mmcv-full
pip install timm==0.3.2 pylint debugpy opencv-python-headless attrs ipython tqdm imageio scikit-image omegaconf
pip install mmcv-full==1.2.7 --no-cache-dir

#-- Clone and install segformer
git clone https://github.com/NVlabs/SegFormer
cd SegFormer
pip install .
```

Download the pretrained model `segformer.b5.1024x1024.city.160k.pth` from the google_drive / one_drive links in https://github.com/NVlabs/SegFormer#evaluation .

Remember the location where you download into, and pass it to the script in the next step with `--checkpoint` .

#### > Extract masks

:pushpin: For [StreetSurf](https://ventusff.github.io/streetsurf_web/) paper: additionally specify `--seq_list=waymo_static_32.lst`

```shell
cd /path/to/neuralsim/dataio/autonomous_driving/waymo

# Extract masks
python extract_masks.py --data_root=/path/to/waymo/processed --segformer_path=/path/to/SegFormer/ --checkpoint=/path/to/SegFormer/pretrained/segformer.b5.1024x1024.city.160k.pth --seq_list=waymo_static_32.lst
```

NOTE: You can pass `--verbose` and `--ignore_existing` if needed.

## v2 API (TODO)

## DEBUG

We have developed a visualization tool based on the awesome library [vedo](https://vedo.embl.es/).

To try it out:

```shell
cd /path/to/neuralsim
source set_env.sh
python dataio/autonomous_driving/waymo/waymo_dataset.py
```

Some example screen recordings:

https://github.com/PJLab-ADG/neuralsim/assets/25529198/b502d393-dcf1-4a87-ba0f-bfa7eae26eee

https://github.com/PJLab-ADG/neuralsim/assets/25529198/42080c6e-6bde-4ecf-8d42-55b5fddf7be9
