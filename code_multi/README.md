## Major settings

For multi-object datasets. FG stands for foreground, and BG stands for background.

| Methods                                                      | Official / Un-official | Get started                        | Notes, major difference from paper, etc.                     |
| ------------------------------------------------------------ | ---------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| Neuralsim | Official               | [readme](../docs/methods/neuralsim.md) |      |

## General guide

```shell
python code_multi/tools/run.py train,render --config code_multi/configs/exps/fg_neus=permuto/all_occ.with_normals.240201.yaml --render.downscale=4
```