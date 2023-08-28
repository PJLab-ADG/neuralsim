
# [neuralsim] COLMAP-like dataset

This DatasetIO can be used for any dataset collected with COLMAP. 

The directory file structure should be as follows:

```
data_dir
├── images
│   ├── 00000000.jpg
│   ├── 00000001.jpg
│   ├── ...
└── sparse
    └── 0
        ├── cameras.bin (or cameras.txt)
        ├── images.bin (or images.txt)
        └── points3D.bin
```

## Extract monocular normals & depths priors

NOTE:

- Normal cues are generally more important than depth cues. In most cases, using only normal cues is sufficient.
- The scale and shift of monocular depths have no correlation with real-world depths. They can only be indirectly used as weak hints.
- The inferred normals are in range `[-1,1]`. The inferred depths are typically in range `[0,1]`.

### Setup `omnidata`

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

### Extract mono cues

```shell
cd /path/to/neuralsim/dataio/autonomous_driving/waymo

# Extract depth
python extract_mono_cues.py --task=depth --data_dir=/path/to/data_dir --omnidata_path=/path/to/omnidata/omnidata_tools/torch/

# Extract normals
python extract_mono_cues.py --task=normal --data_dir=/path/to/data_dir --omnidata_path=/path/to/omnidata/omnidata_tools/torch/
```

NOTE: You can pass `--verbose` if needed.