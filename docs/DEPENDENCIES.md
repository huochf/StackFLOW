# Dependencies (For Post-Optimization)

## EPro-Pnp

```
cd $PROJECT_DIR
mkdir externals && cd ./externals
git clone https://github.com/tjiiv-cprg/EPro-PnP.git
```

you may need to fix this bug to make it compatible with the higher version of the Pytorch by changing  `torch.solve(b, A)[0]` to  `torch.linalg.solve(A, b)[0]` in file `EPro-PnP/EPro-PnP-6DoF/lib/ops/pnp/levenberg_marquardt.py` (line 17)

#### Train

Before training, make sure you have downloaded `VOC 2012` dataset following [this instruction](./DATA_PRE.md) and redirect the path `cfg.dataset.bg_dir` to `VOC_DIR` in file `PROJECT_DIR/scripts/train_epro_pnp.py`(line 815).

##### Train EPro-PnP on BEHAVE dataset:

```
python ./scripts/train_epro_pnp.py --root_dir BEHAVE_ROOT_DIR --is_behave
```

##### Train EPro-PnP on InterCap dataset:

```
python ./scripts/train_epro_pnp.py --root_dir INTERCAP_ROOT_DIR
```

##### Train EPro-PnP on BEHAVE-Extended dataset:

```
python ./scripts/train_epro_pnp.py --root_dir BEHAVE_ROOT_DIR --behave_extend
```

The logs and checkpoints will be saved to `PROJECT_DIR/outputs/epro_pnp`.

#### Inference

Run the following scripts to generate 2D-3D corresponding maps for objects.

```
python ./scripts/inference_epro_pnp --root_dir BEHAVE_ROOT_DIR --is_behave
python ./scripts/inference_epro_pnp --root_dir INTERCAP_ROOT_DIR
python ./scripts/inference_epro_pnp --root_dir BEHAVE_ROOT_DIR --behave_extend
```

The corresponding maps will be saved to directories `BEHAVE_ROOT_DIR/epro_pnp`.

We have provide the [pretrained models]([epro_pnp](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/EoenpT0XvEdLocqfi6KO9gEBlPlC1yLIk_fu1lLWcxrwmg?e=Gqelyg)) and [2D-3D corresponding maps](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/Eh2EeVK3wzlOnjUGLIJfc7kBmRnb3UZZXPX4ff0Ev2S9Xg?e=2c6Xfn) in one drive.

## OpenPose

We use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract the keypoints of the human. You can download these keypoints we have generated in [One Drive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/Eh2EeVK3wzlOnjUGLIJfc7kBmRnb3UZZXPX4ff0Ev2S9Xg?e=2c6Xfn).

