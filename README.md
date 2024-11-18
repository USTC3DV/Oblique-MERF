# Oblique-MERF

[Project page](https://ustc3dv.github.io/Oblique-MERF/)&nbsp;  [Paper](https://arxiv.org/abs/2404.09531) 

## Installation

### Create environment

We recommend using conda to manage dependencies. Make sure to install Conda before proceeding.

```shell
conda create --name oblique-merf -y python=3.8
conda activate oblique-merf
```

### Dependencies

Install PyTorch with CUDA (this repo has been tested with CUDA 11.8) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
`cuda-toolkit` is required for building `tiny-cuda-nn`.

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Install Nerfstudio and Oblique-MERF

```bash
cd nerfstudio
pip install -e .

cd ..
pip install -e .
```

## Data preparation

Our data format requirements follow the instant-ngp convention.

### Matrix City

To download the Matrix City dataset, visit the [official page](https://city-super.github.io/matrixcity/). You can opt to download the "small city"  dataset to test your algorithm. This dataset follows the instant-ngp convention, so no preprocessing is required.

### Custom Data

We highly recommend using [Metashape](https://www.agisoft.com/) to obtain camera poses from multi-view images. Then, use their [script](https://github.com/agisoft-llc/metashape-scripts/blob/master/src/export_for_gaussian_splatting.py) to convert camera poses to the COLMAP convention. Alternatively, you can use [COLMAP](https://github.com/colmap/colmap) to obtain the camera poses. After obtaining the data in COLMAP format, use `ns-process-data` to generate the `transforms.json` file.

```
ns-process-data images --data path/to/data --skip-colmap 
```

Our  loaders expect the following dataset structure in the source path location:

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse(optionally)
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
|---transforms.json
```

## Workflow

### Training

```
ns-trian merf --data path/to/data
```

### Evalation

```
ns-eval --load_config path/to/config --render_output_path path/to/renders --output-path path/to/metrics
```

### Baking 

```
ns-baking --load-config path/to/output/config --baking_config.baking_path path/to/bakings
```

### Real-Time Rendering 


## Citation

If you use this repo or find the documentation useful for your research, please consider citing:

```
@article{zeng2024oblique,
    title={Oblique-MERF: Revisiting and Improving MERF for Oblique Photography},
    author={Zeng, Xiaoyi and Song, Kaiwen and Yang, Leyuan and Deng, Bailin and Zhang, Juyong},
    journal={arXiv preprint arXiv:2404.09531},
    year={2024}
}

```

## Ackownledgements

This repository's code is based on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio), [MERF](https://github.com/google-research/google-research/tree/master/merf) and [City-on-Web](https://github.com/USTC3DV/MERFStudio/tree/main). We are very grateful for their outstanding work.