### Preprocess

We have standardized the preprocessed datasets into **universal formats applicable to all autonomous driving datasets**. For more details on this universal format, please refer to [docs/data/autonomous_driving.md](../../../docs/data/autonomous_driving.md). Regardless of the type of autonomous driving dataset you are using, once converted into this preprocessed format, it can be directly loaded by [custom_autodrive_dataset.py](custom_autodrive_dataset.py).

### Prior extraction - monocular cues and masks

For extraction of monocular depths and surface normals cues, please refer to [Extract monocular normals & depths priors](../waymo/README.md#extract-monocular-normals--depths-priors). 

For extraction of semantic masks for sky and dynamic objects, please refer to [Extract mask priors -  for sky, pedestrian etc](../waymo/README.md#extract-mask-priors----for-sky-pedestrian-etc).

