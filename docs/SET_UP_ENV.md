# Setup Environments

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

#### 4. Install [[Neural Renderer]](https://github.com/JiangWenPL/multiperson/tree/master) (For Visualization Only)

```
cd $PROJECT_DIR/externals
git clone https://github.com/JiangWenPL/multiperson.git
```

add

```
#ifndef AT_CHECK 
#define AT_CHECK TORCH_CHECK 
#endif 
```

at the beginning of files `multiperson/neural_renderer/neural_renderer/cuda/*.cpp`.

```
export CUDA_HOME=/public/software/CUDA/cuda-11.3/ (modify this path to you custom setting)
cd multiperson/neural_renderer && python setup.py install
```

