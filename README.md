# MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes

This repository contains the official implementation for [MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes](https://arxiv.org/abs/2412.11457)

### [Project Page](https://jason-aplp.github.io/MOVIS/)  | [Paper](https://arxiv.org/abs/2412.11457) | [Weights](https://huggingface.co/datasets/JasonAplp/MOVIS/blob/main/last.ckpt) | [Dataset](https://huggingface.co/datasets/JasonAplp/MOVIS/tree/main)

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
Revise the parameters within the script accordingly if you want to change example.
We use [SAM](https://github.com/facebookresearch/segment-anything) and [Depth-FM](https://github.com/CompVis/depth-fm) for getting estimated mask and depth. Notably, you should crop out the background area in the depth map.

## Dataset inference

```bash
bash eval_single.sh
```

## Training

```bash
bash eval_single.sh
```