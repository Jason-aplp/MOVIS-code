# MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes

This repository contains the official implementation for [MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes](https://arxiv.org/abs/2412.11457)

### [Project Page](https://jason-aplp.github.io/MOVIS/)  | [Paper](https://arxiv.org/abs/2412.11457) | [Weights](https://huggingface.co/datasets/JasonAplp/MOVIS/blob/main/last.ckpt) | [Dataset](https://huggingface.co/datasets/JasonAplp/MOVIS/tree/main) | [Rendering_Scripts](https://github.com/Jason-aplp/MOVIS-render)

## Install

```bash
conda create -n movis python=3.9
conda activate movis
cd MOVIS
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```
Download the checkpoint and put it under `MOVIS`.

## Single-Image inference

```bash
bash eval_single.sh
```
Revise the parameters within the script accordingly if one wants to change example.
We use [SAM](https://github.com/facebookresearch/segment-anything) and [Depth-FM](https://github.com/CompVis/depth-fm) for getting estimated mask and depth. The background area in the depth map should be cropped out.

## Dataset inference
Download [C_Obj](https://huggingface.co/datasets/JasonAplp/MOVIS/tree/main/C_Obj) or [C3DFS_test split](https://huggingface.co/datasets/JasonAplp/MOVIS/tree/main/MOVIS-test) for benchmarking.
```bash
bash eval_batch_3d.sh
bash eval_batch_cobj.sh
```
You should revise the dataset path in the `configs/inference_cobj.yaml` and `configs/inference_c3dfs.yaml` file (data-params-root_dir) before running the training script.

## Training
Download image-conditioned stable diffusion checkpoint released by Lambda Labs:
```bash
wget https://cv.cs.columbia.edu/zero123/assets/sd-image-conditioned-v2.ckpt
```
Download the dataset from [here](https://huggingface.co/datasets/JasonAplp/MOVIS/tree/main), the dataset structure should be like this:
```
MOVIS-train/
    000000_004999/
        0/
        1/
        ...
    095000_099999/
    train_path.json
```
Run training script:
```bash
bash train.sh
```
One should revise the dataset path in the `configs/3d_mix.yaml` file (data-params-root_dir) before running the training script.
Note that this training script is set for an 8-GPU system, each with 80GB of VRAM. If you have smaller GPUs, consider using smaller batch size and gradient accumulation to obtain a similar effective batch size.


## Acknowledgement
This repository is based on [Zero123](https://github.com/jason-aplp/Zero123). We would like to thank the authors of these work for publicly releasing their code.

## Citation
```
@article{lu2024movis,
  title={MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes},
  author={Lu, Ruijie and Chen, Yixin and Ni, Junfeng and Jia, Baoxiong and Liu, Yu and Wan, Diwen and Zeng, Gang and Huang, Siyuan},
  journal={arXiv preprint arXiv:2412.11457},
  year={2024}
}
```