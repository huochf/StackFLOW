## Instructions to Reproduce the Experiments

### Set up Environments

#### 1. Create New Environments

```
conda create -n StackFLOW python=3.9
conda activate StackFLOW
```

#### 2. Install pytorch and pytorch3D

Please follow [[Install Pytorch3D]](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to download the appropriate version for pytorch and pytorch3D. Following instructions work for me.

```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

#### 3. Install Other Libraries

```
cd $PROJECT_DIR
pip install -r /requirements.txt
```

#### 4. Install ProHMR

```
cd $PROJECT_DIR
mkdir externals && cd ./externals
git clone https://github.com/nkolot/ProHMR.git
cd ./ProHMR && python setup.py develop
git clone https://github.com/nkolot/nflows.git 
cd ./nflows && python setup.py install
```

### Prepare Datasets

#### 1. BEHAVE

Go to [[BEHAVE | Real Virtual Humans (mpg.de)]](https://virtualhumans.mpi-inf.mpg.de/behave/license.html) to download:

1. scanned objects
2. calibration files
3. Train and test split
4. All Sequences separated by dates.

You should decompress and organize them according to [[Code to access BEHAVE dataset]](https://github.com/xiexh20/behave-dataset).

If you want to train StackFLOW with augmented data, go to [[Onedrive]](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/Ehb3XsrhnxtMmbziYUPlyKsBOQMdL8BtMTxqUoQXXK8Wrg?e=IylOnn) to download rendered fake images and unzip them to `BEHAVE_ROOT_DIR`.

#### 2. InterCap

Go to [[INTERCAP (mpg.de)]](https://intercap.is.tue.mpg.de/download.php) to log in and download:

1. RGBD_images.zip
2. Res.zip

unzip them, 

Go to [[InterCap at master]](https://github.com/YinghaoHuang91/InterCap/tree/master) to get object template meshes and calibration files by running:

```
git clone -b master https://github.com/YinghaoHuang91/InterCap.git
cd InterCap
mkdir PROJECT_ROOT/data/intercap
cp -r ./obj_track/objs/ PROJECT_ROOT/data/intercap
cp -r ./Data/calibration_third PROJECT_ROOT/data/intercap
```

There may be some wrong with `obj_track/objs/08.ply`, download from this copy, and replace it.

#### 3. Background Images

We follow [[CDPN]](https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi) to augment our training data by changing the background. Please go to [[VOC 2012]](http://host.robots.ox.ac.uk/pascal/VOC/index.html) to download the [[background images]](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), unzip them, and put them into the folder `VOC_DIR`.

### Preprocess Datasets

#### 1. Preprocess BEHAVE

Run scripts `$PROJECT_DIR/stackflow/preprocess/preprocess_behave.py` to generate training data list and extract PCA models. After running this scripts, you will get these files:

1. `$PROJECT_DIR/data/behave/behave_real_train_data_list.pkl`
2. `$PROJECT_DIR/data/behave/behave_real_test_data_list.pkl`
3. `$PROJECT_DIR/data/behave/behave_fake_data_list.pkl`
4. `$PROJECT_DIR/data/behave/behave_pca_models/pca_models_n32_d32.pkl`

You can skip this step by downloading these files from [[One Drive]](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/EkDEAudBnVxMknCx0JpO6iEB2Jk13A3KtENLib7XkDP3bA?e=8mt1ui).

#### 2. Preprocess InterCap

Run scripts `$PROJECT_DIR/stackflow/preprocess/preprocess_intercap.py` to generate training data list and extract PCA models. After running this scripts, you should get these files:

1. `$PROJECT_DIR/data/intercap/intercap_data_list_train_seq_wise_split.pkl`
2. `$PROJECT_DIR/data/intercap/intercap_data_list_test_seq_wise_split.pkl`
3. `$PROJECT_DIR/data/intercap/intercap_pca_models/pca_models_n32_d32_seq_wise.pkl`

You can skip this step by downloading these files from [[One Drive]](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/EkDEAudBnVxMknCx0JpO6iEB2Jk13A3KtENLib7XkDP3bA?e=8mt1ui).

#### 3. Download SMPLH and SMPLX models

To train and evaluate the model on BEHAVE dataset and InterCap dataset, you need to download the SMPL + H models and SMPL + X models. Follow [[this instruction]](https://github.com/vchoutas/smplx) to download these models to path `$PROJECT_DIR/data/models/` and organize these files as follows:

```
PROJECT_DIR
├── data
    ├── models
        ├── smplh
        │   ├── SMPLH_FEMALE.pkl
        │   └── SMPLH_MALE.pkl
        └── smplx
            ├── SMPLX_FEMALE.pkl
            ├── SMPLX_MALE.pkl
            └── SMPLX_NEUTRAL.pkl
```

#### 4. Subsampling SMPL mesh (for BSTRO-HOI)

If you want to train BSTRO models on BEHAVE dataset, you should run

```
cd $PROJECT_DIR
python -m stackflow.dataset.behave_hoi_contact_dataset
```

If you want to train BSTRO models on InterCap dataset, you should run

```
cd $PROJECT_DIR
python -m stackflow.dataset.intercap_hoi_contact_dataset
```

These two scripts will generate the subsampled vertices for SMPL mesh and object meshes and write them to the file `$PROJECT_ROOT/data/behave/subsampling_points.pkl` and `PROJECT_ROOT/data/intercap/subsampling_points.pkl` correspondingly.

#### 5. Download Pretrained Params. for HR-Net (for BSTRO-HOI)

```
BLOB='https://datarelease.blob.core.windows.net/metro'
wget -nc $BLOB/models/hrnetv2_w64_imagenet_pretrained.pth -O $PROJECT_DIR/data/hrnet/hrnetv2_w64_imagenet_pretrained.pth
wget -nc $BLOB/models/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml -O $PROJECT_DIR/data/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
```

### Train Models

#### 1. Train StackFLOW on BEHAVE dataset

Before training, you should redirect paths  `BEHAVE_DIR`, `FAKE_IMAGE_DIR` in file `$PROJECT_DIR/stackflow/datasets/behave_metadata.py` and `dataset.bg_dir` in file `PROJECT_DIR/configs/model_behave.yaml` in your custom settings.

```
cd $PROJECT_DIR
python -m stackflow.train --config configs/model_behave.yaml
```

#### 2. Train StackFLOW on InterCap dataset

Before training, you should redirect paths  `INTERCAP_DIR` in file `$PROJECT_DIR/stackflow/datasets/intercap_metadata.py` and `dataset.bg_dir` in file `$PROJECT_DIR/configs/model_intercap.yaml` in your custom settings.

```
cd $PROJECT_DIR
python -m stackflow.train --config configs/model_intercap.yaml
```

#### 3. Train BSTRO-HOI

```
cd $PROJECT_DIR
python -m stackflow.train_bstro_hoi --config configs/bstro_hoi_behave.yaml
(python -m stackflow.train_bstro_hoi --config configs/bstro_hoi_intercap.yaml)
```

We have provided the [pretrained models](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/EtE10I5tSjBJkYAT_3djn_YBpQQm0ud8PNhqVvM5KrPItQ?e=LZFqbx) in one drive.

### Evaluate Models

#### 1. Evaluate StackFLOW

Before you evaluate StackFLOW, you need to use [[Openpose(lightweight)]](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) and [[EPro-PnP]](https://github.com/tjiiv-cprg/EPro-PnP) to generate body keypoints for the person and the 2D-3D corresponding maps for the object. Here, we provide them (person_keypoints_test.json and epro_pnp\_*_recon_results.pk) which are used for our experiments in [[One Drive]](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/EkDEAudBnVxMknCx0JpO6iEB2Jk13A3KtENLib7XkDP3bA?e=8mt1ui). Please download them and put them in the folder `$PROJECT_DIR/data`.  Make sure the model checkpoints have been saved to `$PROJECT_DIR/outputs/stackflow_*/stackflow_*.pth`. Then run

```
cd $PROJECT_DIR
python -m stackflow.joint_optimize_hoi --config configs/model_behave.yaml
(python -m stackflow.joint_optimize_hoi --config configs/model_intercap.yaml)
```

The reconstruction results and evaluation metrics will be saved to the directory `$PROJECT_DIR/outputs/stackflow_*/`.

#### 2. Evaluate BSTRO-HOI

Before you evaluate BSTRO-HOI, you need to use `$PROJECT_DIR/stackflow/train_bstro_hoi.py` to generate contact points. The contact points generated using BSTRO-HOI models in our experiments can be found in [[One Drive]]([outputs](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/huochf_shanghaitech_edu_cn/EtE10I5tSjBJkYAT_3djn_YBpQQm0ud8PNhqVvM5KrPItQ?e=dv7AKR)).  Download them and put them into the folder `$PROJECT_DIR/outputs/bstro_hoi`. Then run

```
cd $PROJECT_DIR
python -m stackflow.bstro_optimization --config configs/prohmr_behave.yaml
(python -m stackflow.bstro_optimization --config configs/intercap_behave.yaml)
```

The reconstruction results and evaluation metrics will be saved to the directory `$PROJECT_DIR/outputs/prohmr_*/`.

## Acknowledgments

This work borrows some codes from [[ProHMR]](https://github.com/nkolot/ProHMR) and [[CDPN]](https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi). Thanks for these fantastic works.