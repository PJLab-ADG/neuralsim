# neuralsim

3D surface reconstruction and simulation based on 3D neural rendering.

This repository primarily addresses two topics:

- Efficient and detailed reconstruction of implicit surfaces across different scenarios.
  - Including object-centric / street-view, indoor / outdoor, large-scale (WIP) and multi-object (WIP) datasets.
  - Highlighted implementations include [neus_in_minutes](docs/methods/neus_in_minutes.md),   [neus_in_minutes#indoor](docs/methods/neus_in_minutes.md#indoor-datasets) and [streetsurf](docs/methods/streetsurf.md).
- Multi-object implicit surface reconstruction, manipulation, and multi-modal sensor simulation.
  - With particular focus on autonomous driving datasets.

**TOC**

- [Implicit surface is all you need !](#implicit-surface-is-all-you-need-)
- [Ecosystem](#ecosystem)
  - [Highlighted implementations](#highlighted-implementations)
- [Highlights](#highlights)
  - [:hammer_and_wrench: Multi-object volume rendering](#hammer_and_wrench-multi-object-volume-rendering)
  - [:bank: Editable assetbank](#bank-editable-assetbank)
  - [:camera: Multi-modal sensor simulation](#camera-multi-modal-sensor-simulation)
- [Usage](#usage)
  - [Installation](#installation)
  - [`code_single` Single scene](#code_single-single-scene)
  - [`code_multi` Multi-object scene](#code_multi-multi-object-scene)
  - [`code_large` Large-scale scene](#code_large-large-scale-scene)
- [Roadmap \& TODOs](#roadmap--todos)
- [Acknowledgements \& citations](#acknowledgements--citations)

## Implicit surface is all you need !

Single-object / multi-object / indoor / outdoor / large-scale surface reconstruction and multi-modal sensor simulation

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| :rocket: Object **surface reconstruction** in minutes !<br />Input: posed images <u>without mask</u><br />Get started: [neus_in_minutes](docs/methods/neus_in_minutes.md)<br />Credits: [Jianfei Guo](https://github.com/ventusff)<br /><img src="https://github.com/PJLab-ADG/neuralsim/assets/25529198/618ff7ab-d769-47c4-9c26-bc54674e0cb2" alt="teaser_training_bmvs_gundam" width="320"> | :rocket:  Outdoor **surface reconstruction** in minutes !<br />Input: posed images <u>without mask</u><br />Get started: [neus_in_minutes](docs/methods/neus_in_minutes.md)<br />Credits: [Jianfei Guo](https://github.com/ventusff)<br /><img src="https://github.com/PJLab-ADG/neuralsim/assets/25529198/7c216407-0c60-47ae-b1c1-f3ba66c67cc2" alt="teaser_training_bmvs_village_house" width="320"> |
| :rocket: Indoor **surface reconstruction** in minutes !<br />Input: posed images, monocular cues<br />Get started: [neus_in_minutes#indoor](docs/methods/neus_in_minutes.md#indoor-datasets)<br />Credits: [Jianfei Guo](https://github.com/ventusff)<br /><img src="https://github.com/PJLab-ADG/neuralsim/assets/25529198/a8ab59a3-c6e2-464e-aeda-0684bf18dbb6" width="320"> | :car: Categorical **surface reconstruction** in the wild !<br />Input: multi-instance multi-view categorical images<br />[To be released 2023.09]<br />Credits: [Qiusheng Huang](https://github.com/huangqiusheng), [Jianfei Guo](https://github.com/ventusff), [Xinyang Li](https://github.com/imlixinyang)<br /><img src="https://github.com/PJLab-ADG/neuralsim/assets/25529198/b557570f-0f88-4c55-b46d-a6d5cf7f4e45" width="320"> |
| :motorway: Street-view **surface reconstruction** in 2 hours !<br />Input: posed images, monocular cues (and optional LiDAR)<br />Get started: [streetsurf](docs/methods/streetsurf.md)<br />Credits: [Jianfei Guo](https://github.com/ventusff), [Nianchen Deng](https://github.com/dengnianchen) <br /> <video src="https://github.com/PJLab-ADG/neuralsim/assets/25529198/7ddfe845-013c-479a-81a6-066bd04cf97c"></video>(Refresh if video won't play) | :motorway: Street-view multi-modal **sensor simulation** ! <br />Using reconstructed asset-bank<br />Get started: [streetsurf#lidarsim](docs/methods/streetsurf.md#lidar-simulation)<br />Credits: [Jianfei Guo](https://github.com/ventusff), [Xinyu Cai](https://github.com/HueyTsai), [Nianchen Deng](https://github.com/dengnianchen) <br/> <video src="https://github.com/PJLab-ADG/neuralsim/assets/25529198/fa6e37e1-48eb-434a-89e8-9b1c230ce50d"></video>(Refresh if video won't play) |
| :motorway: Street-view multi-object **surfaces reconstruction** in hours !<br />Input: posed images, LiDAR, 3D tracklets<br />[To be released 2023.09]<br />Credits: [Jianfei Guo](https://github.com/ventusff), [Nianchen Deng](https://github.com/dengnianchen)<br /><video src="https://github.com/PJLab-ADG/neuralsim/assets/25529198/e2f0b02e-86f0-4c06-85ce-58980e6bbf96"></video>(Refresh if video won't play) | :motorway: Street-view **scenario editing** !<br />Using reconstructed asset-bank<br/>[To be released 2023.09] <br/>Credits: [Jianfei Guo](https://github.com/ventusff), [Nianchen Deng](https://github.com/dengnianchen) <video src="https://github.com/PJLab-ADG/neuralsim/assets/25529198/c130ab03-7b8a-4068-8c2f-9c5bea219e6d"></video>(Refresh if video won't play) |
| :cityscape: Large-scale multi-view surface reconstruction ... (WIP) | :motorway: Street-view light editing ... (WIP)               |

## Ecosystem

```mermaid
%%{init: {'theme': 'neutral', "flowchart" : { "curve" : "basis" } } }%%
graph LR;
    0("fa:fa-wrench <b>Basic models & operators</b><br/>(e.g. LoTD & pack_ops)<br/><a href='https://github.com/pjlab-ADG/nr3d_lib' target='_blank'>nr3d_lib</a>")
    A("fa:fa-road <b>Single scene</b><br/>[paper] StreetSurf<br/>[repo] <a href='https://github.com/pjlab-ADG/neuralsim' target='_blank'>neuralsim</a>/code_single")
    B("fa:fa-car <b>Categorical objects</b><br/>[paper] CatRecon<br/>[repo] <a href='https://github.com/pjlab-ADG/neuralgen' target='_blank'>neuralgen</a>")
    C("fa:fa-globe <b>Large scale scene</b><br/>[repo] neuralsim/code_large<br/>[release date] Sept. 2023")
    D("fa:fa-sitemap <b>Multi-object scene</b><br/>[repo] neuralsim/code_multi<br/>[release date] Sept. 2023")
    B --> D
    A --> D
    A --> C
    C --> D
```

Pull requests and collaborations are warmly welcomed :hugs:! Please follow our code style if you want to make any contribution.

Feel free to open an issue or contact [Jianfei Guo](https://github.com/ventusff) (guojianfei@pjlab.org.cn)  or [Nianchen Deng](https://github.com/dengnianchen) (dengnianchen@pjlab.org.cn) if you have any questions or proposals.

### Highlighted implementations

| Methods                                                      | :rocket: Get started | Official / Un-official                       | Notes, major difference from paper, etc.                     |
| ------------------------------------------------------------ | --------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| [StreetSurf](https://ventusff.github.io/streetsurf_web/)     | [readme](docs/methods/streetsurf.md) | Official                                     | - LiDAR loss improved     |
| [NeuS](https://lingjie0206.github.io/papers/NeuS/) in minutes | [readme](docs/methods/neus_in_minutes.md) | Un-official                                  | - support object-centric datasets as well as <u>indoor</u> datasets<br />- fast and stable convergence without needing mask<br />- support using [NGP](https://github.com/NVlabs/instant-ngp) / [LoTD](https://github.com/pjlab-ADG/nr3d_lib#pushpin-lotd-levels-of-tensorial-decomposition-) or MLPs as fg&bg representations<br />- large pixel batch size (4096) & pixel error maps |
| [NGP](https://github.com/NVlabs/instant-ngp) with LiDAR      | [readme](docs/methods/ngp_lidar.md) | Un-official                                  | - using [Urban-NeRF](https://urban-radiance-fields.github.io/)'s LiDAR loss |
| Multi-object reconstruction with [unisim](https://waabi.ai/unisim/)'s CNN decoder | [WIP] | Un-official<br />:warning: Largely different | - :warning: only the CNN decoder part is similar to [unisim](https://waabi.ai/unisim/) <br />- volumetric ray buffer mering, instead of feature grid spatial merging<br />- our version of foreground hypernetworks and background model [StreetSurf](https://ventusff.github.io/streetsurf_web/) (the details of theirs are not released up to now) |

## Highlights

### :hammer_and_wrench: Multi-object volume rendering

Code: [app/renderers/general_volume_renderer.py](app/renderers/general_volume_renderer.py)

#### > Efficient and universal

We provide a universal implementation of multi-object volume rendering that supports any kind of methods built for volume rendering, as long as a model can be queried with rays and can output `opacity_alpha`, depth samples `t`, and other optional fields like `rgb`, `nablas`, `features`, etc.

This renderer is efficient mainly due to:

- Frustum culling
- Occupancy-grid-based single / batched ray marching and pack merging implemented with [pack_ops](https://github.com/pjlab-ADG/nr3d_lib#pushpin-pack_ops-pack-wise-operations-for-packed-tensors-)
- (optional) Batched / indiced inference of [LoTD](https://github.com/pjlab-ADG/nr3d_lib#pushpin-lotd-levels-of-tensorial-decomposition-)

The figure below depicts the idea of the whole rendering process.

![multi_object_volume_render](media/multi_object_volume_render.png)

#### > Scene graph structure

Code: [app/resources/scenes.py](app/resources/scenes.py)  [app/resources/nodes.py](app/resources/nodes.py)

To streamline the organization of assets and transformations, we adopt the concept of generic scene graphs used in modern graphics engines like [magnum](https://doc.magnum.graphics/magnum/scenegraph.html).

Any entity that possesses a pose or position is considered a node. Certain nodes are equipped with special functionalities, such as camera operations or drawable models (i.e. renderable assets in `AssetBank`).

![scene_graph](media/scene_graph.png)

| Real-data scene graph                          | Real-data frustum culling                              |
| ---------------------------------------------- | ------------------------------------------------------ |
| ![vis_scene_graph](media/vis_scene_graph.jpeg) | ![vis_frustum_culling](media/vis_frustum_culling.jpeg) |

### :bank: Editable assetbank

Code: `code_multi/tools/manipulate.py` (WIP)

Given that different objects are represented by unique networks (for categorical or shared models, they have unique latents or embeddings), it's possible to explicitly add, remove or modify the reconstructed assets in a scene.

We offer a toolkit for performing such scene manipulations. Some of the intriguing edits are showcased below.

| :dancer: Let them dance !                                    | :twisted_rightwards_arrows: Multi-verse                      | :art: Change their style !                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <video src="https://github.com/PJLab-ADG/neuralsim/assets/25529198/c130ab03-7b8a-4068-8c2f-9c5bea219e6d" alt="teaser_seg767010_manipulate"></video>(Refresh if video won't play) | <video src="https://github.com/PJLab-ADG/neuralsim/assets/25529198/13ed4e86-bf20-45e1-97bb-55a23645da97" alt="teaser_seg767010_multiverse"></video>(Refresh if video won't play) | <video src="https://github.com/PJLab-ADG/neuralsim/assets/25529198/206e8024-f5f9-40ca-85fe-19841fb901b7" alt="teaser_seg767010_style"></video>(Refresh if video won't play)<br />Credits to [Qiusheng Huang](https://github.com/huangqiusheng) and [Xinyang Li](https://github.com/imlixinyang). |

Please note, this toolkit is currently in its **early development stages** and only basic edits have been released. Stay tuned for updates, and contributions are always welcome :) 

### :camera: Multi-modal sensor simulation

#### > LiDARs

Code: [app/resources/observers/lidars.py](app/resources/observers/lidars.py)

Get started: 

- [streetsurf#lidar-simulation](docs/methods/streetsurf.md#lidar-simulation)

Credits to [Xinyu Cai's team work](https://github.com/PJLab-ADG/LiDARSimLib-and-Placement-Evaluation), we now support simulation of various real-world LiDAR models. 

The volume rendering process is guided by our reconstructed implicit surface scene geometry, which guarantees accurate depths. More details on this are in our [StreetSurf](https://ventusff.github.io/streetsurf_web/) paper section 5.1.

#### > Cameras

Code: [app/resources/observers/cameras.py](app/resources/observers/cameras.py)

We now support pinhole camera, standard OpenCV camera models with distortion, and an experimental fisheye camera model.

## Usage

### Installation

First, clone with submodules: 

```shell
git clone https://github.com/pjlab-ADG/neuralsim --recurse-submodules -j8 ...
```

Then, `cd` into `nr3d_lib` and refer to [nr3d_lib/README.md](https://github.com/PJLab-ADG/nr3d_lib#installation) for the following steps.

### `code_single` Single scene

- Object-centric scenarios (indoor / outdoor, with / without mask)
- Street-view or autonomous driving scenarios

Please refer to [code_single/README.md](code_single/README.md)

### `code_multi` Multi-object scene

(WIP)

### `code_large` Large-scale scene

(WIP)

## Roadmap & TODOs

- [ ] Unofficial implementation of unisim
- [ ] Release our methods on multi-object reconstruction for autonomous driving
- [ ] Release our methods on large-scale representation and neus
- [ ] Factorization of embient light and object textures
- [ ] Dataloaders for more autonomous driving datasets (KITTI, NuScenes, Waymo v2.0, ZOD, PandarSet)

## Acknowledgements & citations

- [nr3d_lib](https://github.com/pjlab-ADG/nr3d_lib)  Containing most of our basic modules and operators 
- [LiDARSimLib](https://github.com/PJLab-ADG/LiDARSimLib-and-Placement-Evaluation)   LiDAR models
- [StreetSurf](https://ventusff.github.io/streetsurf_web/)  Our recent paper studying street-view implicit surface reconstruction

```bibtex
@article{guo2023streetsurf,
  title = {StreetSurf: Extending Multi-view Implicit Surface Reconstruction to Street Views},
  author = {Guo, Jianfei and Deng, Nianchen and Li, Xinyang and Bai, Yeqi and Shi, Botian and Wang, Chiyu and Ding, Chenjing and Wang, Dongliang and Li, Yikang},
  journal = {arXiv preprint arXiv:2306.04988},
  year = {2023}
}
```

- [NeuS](https://lingjie0206.github.io/papers/NeuS/)   Most of our methods are derived from NeuS

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

