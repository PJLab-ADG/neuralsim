# [neuralsim] NeuS in minutes

[website](https://lingjie0206.github.io/papers/NeuS/) | [arxiv](https://arxiv.org/abs/2106.10689) | [official_repo](https://github.com/Totoro97/NeuS) | 

An **unofficial** and improved implementation of "NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction".

```bibtex
@inproceedings{wang2021neus,
	title={NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction},
	author={Wang, Peng and Liu, Lingjie and Liu, Yuan and Theobalt, Christian and Komura, Taku and Wang, Wenping},
	booktitle={Proc. Advances in Neural Information Processing Systems (NeurIPS)},
	volume={34},
	pages={27171--27183},
	year={2021}
}
```

## Highlights (demo coming soon!)

- Stable training within 10 minutes without necessarily needing mask
- Worried about your camera pose accuracy ? We can refine them !
- Worried about your footage quality & consistency ? We have in the wild image embeddings !
- Worried about geometric distortions like depressions or bulges (凹坑或鼓包) ? We opt to use monocular normal priors !
- Object-centric, indoor or outdoors ? We can cover them all !

## Object-centric datasets

### Requirements

- <10 mins training time on single RTX3090
- 6 GiB GPU Mem

### Major settings

| Settings                                            | Dataset                                                      | Config file                                                  |
| --------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Multi-view reconstruction without mask (BlendedMVS) | [BlendedMVS dataset preparation](../../dataio/bmvs/README.md) | [lotd_neus.bmvs.230814.yaml](../../code_single/configs/object_centric/lotd_neus.bmvs.230814.yaml) |
| Multi-view reconstruction with mask (DTU)           | [NeuS/DTU dataset preparation](../../dataio/dtu/README.md)   |                                                              |
| Multi-view reconstruction without mask (DTU)        | [NeuS/DTU dataset preparation](../../dataio/dtu/README.md)   |                                                              |

### Instructions

For detailed instructions, please refer to the [general guide](../../code_single/README.md#general-usage) section in `code_single`.

## Indoor datasets

Can be viewed as an **unofficial** and improved implementation of "MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction".

[website](https://niujinshuchong.github.io/monosdf/) | [arxiv](https://arxiv.org/abs/2206.00665) | [offcial_repo](https://github.com/autonomousvision/monosdf) | :warning: Unofficial implementation :warning:

```bibtex
@inproceedings{Yu2022MonoSDF,
	author = {Yu, Zehao and Peng, Songyou and Niemeyer, Michael and Sattler, Torsten and Geiger, Andreas}, 
	title = {MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction}, 
	booktitle={Proc. Advances in Neural Information Processing Systems (NeurIPS)}, 
	year = {2022}, 
}
```

### Requirements

- <10 mins training time on single RTX3090
- 6 GiB GPU Mem

### Dataset preparation

Follow [this link](https://github.com/autonomousvision/monosdf#dataset) to download the MonoSDF's preprocessed data of [Replica](https://github.com/facebookresearch/Replica-Dataset) / [scannet](http://www.scan-net.org/) indoor datasets. 

### Major settings

| Settings / Dataset                                           | Config file                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Replica dataset (processed by [MonoSDF](https://github.com/autonomousvision/monosdf)) | [lotd_neus.replica.230814.yaml](../../code_single/configs/indoor/lotd_neus.replica.230814.yaml) |
| Scan net dataset (processed by [MonoSDF](https://github.com/autonomousvision/monosdf)) | WIP                                                          |

### Instructions

For detailed instructions, please refer to the [general guide](../../code_single/README.md#general-usage) section in `code_single`.
