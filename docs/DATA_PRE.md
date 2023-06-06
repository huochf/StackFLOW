# Data Preparation

## Download SMPL Parameters

To train and evaluate the model on BEHAVE dataset and InterCap dataset, you need to download the SMPL + H models and SMPL + X models. Follow [this instruction](https://github.com/vchoutas/smplx) to download these models to path `PROJECT_DIR/data/models/` and organize these files as follows:

```
PROJECT_DIR
├── data
    ├── models
        ├── smpl
        │   ├── SMPL_FEMALE.pkl
        │   ├── SMPL_MALE.pkl
        │   └── SMPL_NEUTRAL.pkl
        ├── smplh
        │   ├── SMPLH_FEMALE.pkl
        │   └── SMPLH_MALE.pkl
        └── smplx
            ├── SMPLX_FEMALE.pkl
            ├── SMPLX_MALE.pkl
            └── SMPLX_NEUTRAL.pkl
```

## Prepare BEHAVE Dataset

#### 1. Download Dataset

Go to [BEHAVE | Real Virtual Humans (mpg.de)](https://virtualhumans.mpi-inf.mpg.de/behave/license.html) to download:

1. scanned objects
2. calibration files
3. Train and test split
4. All Sequences separated by dates.

You should decompress and organize them according to [Code to access BEHAVE dataset](https://github.com/xiexh20/behave-dataset).

<font color='red'>You can skip the following step #2 - step #6 by downloading the corresponding files, which we generated during our experiments, in [One Drive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/Eh2EeVK3wzlOnjUGLIJfc7kBmRnb3UZZXPX4ff0Ev2S9Xg?e=hDPJYM) and [One Drive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/EiZB8nYaGgNDt74oEDC9mYIBJeOowX4ui351IYQ0e0DeGQ?e=Tra6ns).</font>

#### 2. Generate Viewport-free Augmented Data (optional)

Use [SCANimate](https://github.com/shunsukesaito/SCANimate) to generate clothed avatars and rend them with objects. For each HOI instance in BHEAVE training set, we have rendered fake images with 12 viewports and 4 avatars. We provide the augmented data in [One Drive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/Ehb3XsrhnxtMmbziYUPlyKsBOQMdL8BtMTxqUoQXXK8Wrg?e=P0bQhh). Download them and unzip them to the path `BEHAVE_ROOT_DIR/rendered_images`.

TODO: In detail, show how you generate these fake images with new viewports.

#### 3. Generate 2D-3D Correspondence Maps for Objects

If you want to train Epro-PnP by yourself, you should generate these correspondence maps by running:

```
python ./scripts/render_obj_coor_maps.py --root_dir BEHAVE_ROOT_DIR --is_behave
```

This script will render and write the correspondence maps and the rendered masks to the directory `BEHAVE_ROOT_DIR/object_coor_maps/`. 

#### 4. Generate Training Data List

```
python ./scripts/preprocess_annotations.py --root_dir BEHAVE_ROOT_DIR --is_behave
```

This script will generate the data lists and write them into file `PROJECT_DIR/data/datasets/behave_train_list.pkl` and file `PROJECT_DIR/data/datasets/behave_test_list.pkl`.

Run the following script to visualize them.

```
python ./scripts/visualize_annotation.py --root_dir BEHAVE_ROOT_DIR --anno_file ./data/datasets/behave_train_list.pkl --is_behave
```

The visualized images will be saved to `PROJECT_DIR/outputs/visualize_anno/behave`.

#### 5. Generate Training Data List for Augmented Data (optional)

```
python ./scripts/preprocess_annotations.py --root_dir BEHAVE_ROOT_DIR --is_behave --for_aug
```

This script will generate the training data list and write it into the file `PROJECT_DIR/data/datasets/behave_aug_data_list.pkl`.

Run the following script to visualize them.

```
python ./scripts/visualize_annotation.py --root_dir BEHAVE_ROOT_DIR --anno_file ./data/datasets/behave_aug_data_list.pkl --is_behave --for_aug
```

The visualized images will be saved to `PROJECT_DIR/outputs/visualize_anno/behave_aug`.

#### 6. Construct PCA Latent Space for HOI Spatial Relation

```
python ./scripts/extract_pca.py --root_dir BEHAVE_ROOT_DIR --is_behave
```

This script will generate and write the PCA models to the path `PROJECT_DIR/data/datasets/behave_pca_models_n32_64_d32.pkl`.

## Prepare InterCap Dataset

#### 1. Download Dataset

Go to [INTERCAP (mpg.de)](https://intercap.is.tue.mpg.de/download.php) to log in and download:

1. RGBD_images.zip
2. Res.zip

unzip them, 

Go to [InterCap at master](https://github.com/YinghaoHuang91/InterCap/tree/master) to get object template meshes and calibration files by running:

```
git clone -b master https://github.com/YinghaoHuang91/InterCap.git
cd InterCap
cp -r ./obj_track/objs/ INTERCAP_ROOT_DIR
cp -r ./Data/ INTERCAP_ROOT_DIR
```

There may be some wrong with `obj_track/objs/08.ply`, download [this copy](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/huochf_shanghaitech_edu_cn/EVJ4Uwyatr5EvvhduGRMvLkBvMiZZs4z5-4uXNw9hgfjfA?e=6K7dlq), and replace it.

<font color='red'>You can skip the following step #2 - step #6 by downloading the corresponding files, which we generated during our experiments, in [One Drive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/EvLO78UnaFxKpUhEHWGiBQoB7veTcsEvVioR8TGwYDNXwQ?e=FNojyF) and [One Drive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/EiZB8nYaGgNDt74oEDC9mYIBJeOowX4ui351IYQ0e0DeGQ?e=Tra6ns).</font>

#### 2. Tune Poses for SMPLX and Objects

We find that annotation provided in INTERCAP_ROOT_DIR/Res/Sub_id/Obj_id/Seq_id/res_*.pkl is not accuracy, we finetuned them by running:

```
python ./scripts/tune_annotations_intercap --root_dir INTERCAP_ROOT_DIR
```

The tuned parameters for smplx and 6D pose for object will be written to folder `INTERCAP_ROOT_DIR/Res_tuned`.

#### 3. Generate 2D-3D Correspondence Maps for Objects

```
python ./scripts/render_obj_coor_maps --root_dir INTERCAP_ROOT_DIR
```

This script will render and write the correspondence maps and the rendered masks to the directory `INTERCAP_ROOT_DIR/object_coor_maps/`. 

#### 4. Extract Person Mask

Go to [PointRend](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend) to download pointrend weights (PointRend x101-FPN 3x) to `data/weights` , then run:

```
python -W ignore ./scripts/extract_person_mask.py --root_dir INTERCAP_ROOT_DIR
```

This script will write the person masks to the directory `INTERCAP_ROOT_DIR/mask`. (Note that this step can be run in parallel with step #3)

#### 5. Generate Training Data List

```
python ./scripts/preprocess_annotations.py --root_dir INTERCAP_ROOT_DIR
```

This script will generate and write the data list into the file `PROJECT_DIR/data/datasets/intercap_train_list.pkl` and the file `PROJECT_DIR/data/dataset/intercap_test_list.pkl`.

Run the following script to visualize them.

```
python ./scripts/visualize_annotation.py --root_dir INTERCAP_ROOT_DIR --anno_file ./data/datasets/intercap_train_list.pkl
```

The visualized images will be saved to `PROJECT_DIR/outputs/visualize_anno/intercap`.

#### 6. Construct PCA Latent Space for HOI Spatial Relation

```
python ./scripts/extract_pca.py --root_dir INTERCAP_ROOT_DIR
```

This script will generate and write the PCA models to the path `PROJECT_DIR/data/datasets/intercap_pca_models_n32_64_d32.pkl`.

## Prepare BEHAVE-extended Dataset

#### 1. Download Dataset

If you want to train the model on the extended BEHAVE dataset, you need also go to [BEHAVE | Real Virtual Humans (mpg.de)](https://virtualhumans.mpi-inf.mpg.de/behave/license.html) to download:

5. All Raw videos (color videos and frame timestamps)
6. SMPL and object parameters.

unzip all raw videos into folder `BEHAVE_ROOT_DIR/raw_videos`, all parameters into folder `BEHAVE_ROOT_DIR/behave-30fps-params-v1`, and use [video2images](https://github.com/xiexh20/behave-dataset/blob/main/tools/video2images.py) to extract these videos into folder `BEHAVE_ROOT_DIR/raw_images`.

<font color='red'>You can skip step #2, step #4, step #5 by downloading the corresponding files, which we generated during our experiments, in [One Drive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/Eh2EeVK3wzlOnjUGLIJfc7kBmRnb3UZZXPX4ff0Ev2S9Xg?e=hDPJYM) and [One Drive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/EiZB8nYaGgNDt74oEDC9mYIBJeOowX4ui351IYQ0e0DeGQ?e=Tra6ns).</font>

#### 2. Generate the Valid Frame List

Some frames may lack annotations, filter them out by running:

```
python ./scripts/filter_annotation_behave.py --root_dir BEHAVE_ROOT_DIR
```

This script will merge de-parted annotations, collect all valid frames and write them to `BEHAVE_ROOT_DIR/behave_extend_valid_frames.pkl`.

#### 3. Generate 2D-3D Correspondence Maps for Objects

If you want to train Epro-PnP by yourself, you should generate these correspondence maps by running:

```
python ./scripts/render_obj_coor_maps --root_dir BEHAVE_ROOT_DIR --behave_extend
```

This script will render and write the correspondence maps and the rendered masks to the directory `BEHAVE_ROOT_DIR/object_coor_maps_extend/`. 

#### 4. Extract Person Mask

Go to [PointRend](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend) to download pointrend weights (PointRend x101-FPN 3x) to `data/weights` , then run:

```
python -W ignore ./scripts/extract_person_mask.py --root_dir BEHAVE_ROOT_DIR --behave_extend
```

This script will write the person masks to the directory `BEHAVE_ROOT_DIR/person_mask`. (Note that this step can be run in parallel with step #2)

#### 5. Generate Training Data List

```
python ./scripts/preprocess_annotations.py --root_dir BEHAVE_ROOT_DIR --behave_extend
```

This script will generate the data list and write them into file `PROJECT_DIR/data/datasets/behave_extend_train_list.pkl` and file `PROJECT_DIR/data/dataset/behave_extend_test_list.pkl`.

Run the following script to visualize them.

```
python ./scripts/visualize_annotation.py --root_dir BEHAVE_ROOT_DIR --anno_file ./data/datasets/behave_extend_train_list.pkl --behave_extend
```

The visualized images will be saved to `PROJECT_DIR/outputs/visualize_anno/behave_extend`.

## Prepare Background Images

We follow [CDPN](https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi) to augment our training data by changing the background. Please go to [VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/index.html) to download the [background images](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), unzip them, and put them into the folder `VOC_DIR`.