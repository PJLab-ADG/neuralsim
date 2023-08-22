# [neuralsim] InstantNGP + UrbanNeRF

 [Instant-NGP](https://github.com/NVlabs/instant-ngp)  |  [UrbanNeRF](https://urban-radiance-fields.github.io/)

:warning: Unofficial implementation :warning:

An unofficial implementation (combination) of "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" and "Urban Radiance Fields"

```bibtex
@article{muller2022instantngp,
	title={Instant neural graphics primitives with a multiresolution hash encoding},
	author={M{\"u}ller, Thomas and Evans, Alex and Schied, Christoph and Keller, Alexander},
	journal={ACM Transactions on Graphics (ToG)},
	volume={41},
	number={4},
	pages={1--15},
	year={2022}
}
```

```bibtex
@inproceedings{rematas2022urban,
	title={Urban radiance fields},
	author={Rematas, Konstantinos and Liu, Andrew and Srinivasan, Pratul P and Barron, Jonathan T and Tagliasacchi, Andrea and Funkhouser, Thomas and Ferrari, Vittorio},
	booktitle={Proc. {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages={12932--12942},
	year={2022}
}
```

## Usage

### Requirements

- ~45 mins training time on single RTX3090

- ~11 GiB GPU Mem
- \>20 GiB CPU Mem (Caching data to speed up)

### Dataset preparation

- Waymo Open Dataset - Perception

  - [README](../../dataio/autonomous_driving/waymo/README.md)

  - split file: [waymo_static_32.lst](../../dataio/autonomous_driving/waymo/waymo_static_32.lst)


### Major settings

| Settings                             | Config file                                                  |
| ------------------------------------ | ------------------------------------------------------------ |
| Multi-view reconstruction with LiDAR | [ngp_withlidar.230814.yaml](../../code_single/configs/waymo/ngp_withlidar.230814.yaml) |

## Instructions

For detailed instructions, please refer to the [general guide](../../code_single/README.md#general-usage) section in `code_single`.
