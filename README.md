## This is the AMD ROCm GPU tested version of the original repo. 

To install ROCm and PyTorch suite of software, please refer: [The Rock releases for rocm and PyTorch](https://github.com/ROCm/TheRock/blob/main/RELEASES.md). To install the AMD ROCm GPU version of the submodules and dependencies for this repo, use the commands below:

```shell
pip install open-clip-torch plyfile jaxtyping typing pathlib
pip install submodules/segment-anything-langsplat --no-build-isolation
pip install --no-build-isolation git+https://github.com/jiagaoxiang/langsplat-rasterization.git 
pip install --no-build-isolation git+https://github.com/amd-wangfan/simple-knn.git@hip_support
pip install opencv-python
```

Below are the commands for running the entire commands in the repo tested on AMD GPUs.

```shell
# Clone LangSplat repo
git clone https://github.com/jiagaoxiang/LangSplat.git --recursive
cd LangSplat/

# Download the LERF_OVS dataset
pip install gdown
gdown --id 1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt --no-check-certificate
# Unzip the downloaded dataset
apt-get update
apt-get install unzip
unzip -q lerf_ovs.zip

# Download SAM model checkpoints
mkdir -p ckpts && cd ckpts && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Preprocess the dataset
dataset_path=lerf_ovs/figurines
python preprocess.py --dataset_path $dataset_path

# Train the autoencoder
cd autoencoder
python train.py \
  --dataset_path $dataset_path \
  --dataset_name figurines \
  --encoder_dims 256 128 64 32 3 \
  --decoder_dims 16 32 64 128 256 256 512 \
  --lr 0.0007 \
  --num_epochs 100

# Get the 3-dims language feature of the scene
python test.py \
  --dataset_path $dataset_path \
  --dataset_name figurines \
  --encoder_dims 256 128 64 32 3 \
  --decoder_dims 16 32 64 128 256 256 512

# Train RGB 3DGS 30000 checkpoint
cd ..
python train.py \
  -s lerf_ovs/figurines \
  -m lerf_ovs/figurines/output/figurines \
  --iterations 30000 \
  --no_include_feature

# Train the LangSplat and render
for level in 1 2 3; do
  # Train the LangSplat (include_feature defaults to True)
  python train.py \
    -s $dataset_path \
    -m output/figurines \
    --start_checkpoint lerf_ovs/figurines/output/figurines_-1/chkpnt30000.pth \
    --feature_level $level
  # Render the LangSplat
  python render.py \
    -s $dataset_path \
    -m output/figurines_${level} \
    --feature_level ${level} \
    --include_feature
done

# Eval
cd eval
pip install matplotlib mediapy
# change gt_folder to this location: ../lerf_ovs/label
bash eval.sh
```
The original READ.me file is below:

---

# [CVPR2024 Highlight] LangSplat: 3D Language Gaussian Splatting 
[Minghan Qin*](https://minghanqin.github.io/), [Wanhua Li*†](https://li-wanhua.github.io/), [Jiawei Zhou*](https://latitudezhou.github.io/), [Haoqian Wang†](https://www.sigs.tsinghua.edu.cn/whq_en/main.htm), [Hanspeter Pfister](https://seas.harvard.edu/person/hanspeter-pfister)<br>(\* indicates equal contribution, † means Co-corresponding author)<br>| [Webpage](https://langsplat.github.io/) | [Full Paper](https://arxiv.org/pdf/2312.16084.pdf) | [Video](https://www.youtube.com/watch?v=XMlyjsei-Es) |<br>
| Preprocessed Dataset | [BaiduWangpan](https://pan.baidu.com/s/1S_cdmN9EFOlCQ3z1GZR3EA?pwd=lfea) | [GoogleDrive](https://drive.google.com/drive/folders/1Icw5AcQkY_2L_k7ddXrGCJ3z4laa4jg5?usp=sharing) |<br>
| Pre-trained Models | [BaiduWangpan](https://pan.baidu.com/s/12L83uEi5KlF9ViAZqp0B4w?pwd=dl22) | [GoogleDrive](https://drive.google.com/drive/folders/1ASFXWOwaXP_aSXV2iMDmEfILaDXQXlrE?usp=sharing) |<br>
| [Datasets](https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view?usp=sharing) |<br>

![Teaser image](assets/teaser.png)

This repository contains the official authors implementation associated with the paper "LangSplat: 3D Language Gaussian Splatting" (CVPR 2024), which can be found [here](https://arxiv.org/pdf/2312.16084.pdf). We further provide the preprocessed datasets 3D-OVS with language feature, as well as pre-trained models. 

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">😊LangSplat Family</h2>
    <pre><code>@inproceedings{qin2024langsplat,
  title={Langsplat: 3d language gaussian splatting},
  author={Qin, Minghan and Li, Wanhua and Zhou, Jiawei and Wang, Haoqian and Pfister, Hanspeter},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20051--20060},
  year={2024}
}</code></pre>

  <p><strong>🎉 We have released LangSplat V2!</strong>  
  The new version significantly improves performance, achieving over <strong>450+ FPS</strong> in rendering.  <a href="https://langsplat-v2.github.io/" target="_blank" style="text-decoration: underline;">[NeurIPS 2025] LangSplat V2</a>
  </p>

  <pre><code>@article{li2025langsplatv2,
  title={LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS},
  author={Li, Wanhua and Zhao, Yujie and Qin, Minghan and Liu, Yang and Cai, Yuanhao and Gan, Chuang and Pfister, Hanspeter},
  journal={arXiv preprint arXiv:2507.07136},
  year={2025}
}</code></pre>

  <p>🎉We also invite everyone to check out our <a href="https://4d-langsplat.github.io/" target="_blank" style="text-decoration: underline;">[CVPR 2025] 4D LangSplat</a>, which is a multimodal, object-wise video prompting approach combined with a status deformable network to learn 4D language fields.
  </p> 

  <pre><code>@inproceedings{li20254d,
  title={4d langsplat: 4d language gaussian splatting via multimodal large language models},
  author={Li, Wanhua and Zhou, Renping and Zhou, Jiawei and Song, Yingwei and Herter, Johannes and Qin, Minghan and Huang, Gao and Pfister, Hanspeter},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={22001--22011},
  year={2025}
}</code></pre>
  
  </div> 
</section>

  
  </div>
</section>

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:minghanqin/LangSplat.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/minghanqin/LangSplat.git --recursive
```

## Overview

The codebase has 3 main components:
- A PyTorch-based optimizer to produce a LangSplat model from SfM datasets with language feature inputs to
- A scene-wise language autoencode to alleviate substantial memory demands imposed by explicit modeling.
- A script to help you turn your own images into optimization-ready SfM data sets with language feature

The components have been tested on Ubuntu Linux 18.04. Instructions for setting up and running each of them are found in the sections below.

## Datasets
In the experiments section of our paper, we primarily utilized two datasets: the 3D-OVS dataset and the LERF dataset.

The 3D-OVS dataset is accessible for download via the following link: [Download 3D-OVS Dataset](https://drive.google.com/drive/folders/1kdV14Gu5nZX6WOPbccG7t7obP_aXkOuC?usp=sharing) .

For the LERF dataset, we have expanded upon its existing collection and also provided the corresponding COLMAP data. These resources can be accessed through this link: [Download Expanded LERF Dataset and COLMAP Data](https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view?usp=sharing).

## Optimizer

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used VS Code)
- CUDA SDK 11 for PyTorch extensions (we used 11.8)
- C++ Compiler and CUDA SDK must be compatible

### Setup

#### Environment Setup

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate langsplat
```

### QuickStart

Download the pretrained model to ```output/```, then simply use

```shell
python render.py -m output/$CASENAME --include_feature
```


## Processing your own Scenes

### Before getting started
Firstly, put your images into the data dir.
```
<dataset_name>
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
```
Secondly, you need to acquire the following dataset format and a pre-trained RGB model follow the [3dgs](https://github.com/graphdeco-inria/gaussian-splatting) repository.

```
<dataset_name>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
|---output
|   |---<dataset_name>
|   |   |---point_cloud/iteration_30000/point_cloud.ply
|   |   |---cameras.json
|   |   |---cfg_args
|   |   |---chkpnt30000.pth
|   |   |---input.ply
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```


### Environment setup.
  Please install [segment-anything-langsplat](https://github.com/minghanqin/segment-anything-langsplat) and download the checkpoints of SAM from [here](https://github.com/facebookresearch/segment-anything) to ```ckpts/```.
### Pipeline
Follow the ```process.sh``` and train LangSplat on your own scenes.
- **Step 1: Generate Language Feature of the Scenes.**
  Put the image data into the "input" directory under the ```<dataset_name>/```, then run the following code.
  ```
  python preprocess.py --dataset_path $dataset_path 
  ```
- **Step 2: Train the Autoencoder and get the lower-dims Feature.**
  ```
  # train the autoencoder
  cd autoencoder
  python train.py --dataset_name $dataset_path --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --output ae_ckpt
  # get the 3-dims language feature of the scene
  python test.py --dataset_name $dataset_path --output
  ```

  Our model expect the following dataset structure in the source path location:
  ```
  <dataset_name>
  |---images
  |   |---<image 0>
  |   |---<image 1>
  |   |---...
  |---language_feature
  |   |---00_f.npy
  |   |---00_s.npy
  |   |---...
  |---language_feature_dim3
  |   |---00_f.npy
  |   |---00_s.npy
  |   |---...
  |---output
  |   |---<dataset_name>
  |   |   |---point_cloud/iteration_30000/point_cloud.ply
  |   |   |---cameras.json
  |   |   |---cfg_args
  |   |   |---chkpnt30000.pth
  |   |   |---input.ply
  |---sparse
      |---0
          |---cameras.bin
          |---images.bin
          |---points3D.bin
  ```
- **Step 3: Train the LangSplat.**
  ```
  python train.py -s dataset_path -m output/${casename} --start_checkpoint $dataset_path/output/$casename/chkpnt30000.pth --feature_level ${level}
  ```
- **Step 4: Render the LangSplat.**
  ```
  python render.py -s dataset_path -m output/${casename} --feature_level ${level}
  ```
- **Step 5: Eval.**
  First, we generate the 3-dim language feature map through Step 4. Subsequently, the decoder elevates the features from 3 dimensions to 512 dimensions. For further operations and detailed explanations, please refer to the [supplementary materials](https://arxiv.org/pdf/2312.16084.pdf). 

  -  3D Object Localization on LERF and 3D Semantic Segmentation on LERF. Our eval code is based on [LERF](https://github.com/kerrj/lerf) and [NerfStudio](https://github.com/nerfstudio-project/nerfstudio), thanks for these impressive open-source projects!
  
      - Please download the [lerf_ovs]((https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view?usp=sharing)) first. 
      
      - Set the ```gt_folder``` as the path to lerf_ovs/label. 
      
      - Make sure finish the **Step 4** before you run the eval code. 
  ```
  cd eval
  sh eval.sh
  ```

## TODO list:
- [x] release the code of the optimizer
- [x] release the code of the autoencoder
- [x] release the code of the segment-anything-langsplat
- [x] update the arxiv link
- [x] release the preprocessed dataset and the pretrained model
- [x] release more preprocessed dataset and the pretrained model (coming soon)
- [x] release the code of the eval

This project is still under development. Please feel free to raise issues or submit pull requests to contribute to our codebase.
