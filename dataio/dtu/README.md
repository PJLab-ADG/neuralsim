# [neuralsim] DTU / IDR dataset

This DatasetIO can be used for any dataset collected using the [NeuS](https://github.com/Totoro97/NeuS) / [IDR](https://github.com/lioryariv/idr) data format.

Currently, the most typical example we have tested is the [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36) and [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset provided by [NeuS](https://github.com/Totoro97/NeuS).

## Usage

### Download

Go to the [NeuS](https://github.com/Totoro97/NeuS) repo and download their data.

## DEBUG

```shell
cd /path/to/neuralsim
source set_env.sh
python dataio/dtu/dtu_dataset.py
```

