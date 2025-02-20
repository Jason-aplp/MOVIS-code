from contextlib import nullcontext
from functools import partial
import argparse, os, sys, datetime, glob, importlib, csv

import math
import fire
import gradio as gr
import numpy as np
import torch
import yaml
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms
from ldm.util import load_and_preprocess, instantiate_from_config
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import random
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from omegaconf import DictConfig, ListConfig
import json

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import cv2
import einops

# Create the parser
parser = argparse.ArgumentParser(description="parse argument.")

# Add arguments
parser.add_argument("--input_image", type=str, help="path to input image")
parser.add_argument("--input_depth", type=str, help="path to input depth")
parser.add_argument("--input_mask", type=str, help="path to input mask")
parser.add_argument("--azimuth", type=int, help="azimuth change")
parser.add_argument("--elevation", type=int, help="elevation change")

# Parse the arguments
args = parser.parse_args()

from kiui.op import safe_normalize
def look_at(campos, target, opengl=True):
    """construct pose rotation matrix by look-at.

    Args:
        campos (np.ndarray): camera position, float [3]
        target (np.ndarray): look at target, float [3]
        opengl (bool, optional): whether use opengl camera convention (forward direction is target --> camera). Defaults to True.

    Returns:
        np.ndarray: the camera pose rotation matrix, float [3, 3], normalized.
    """
   
    if not opengl:
        # forward is camera --> target
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # forward is target --> camera
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 0, 1], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model

def get_T2(target_RT, cond_RT):
    delta = torch.from_numpy(np.linalg.inv(cond_RT) @ target_RT).flatten()
    return delta


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, \
                 ddim_eta, T, depth1, mask_super):
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            # c = model.get_learned_conditioning(input_im).tile(1, n_samples,1,1)
            # # T = torch.tensor([math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), z])
            # T = T[:, None, None, :].repeat(1, n_samples, 1, 1).to(c.device)
            c = model.get_learned_conditioning(input_im)
            T = T[:, None, :].to(c.device)
            T = einops.repeat(T, 'b l n -> b (l k) n', k=c.shape[1])
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c.float())
            cond = {}
            cond['c_crossattn'] = [c]
            c_concat = model.encode_first_stage((input_im.to(c.device))).mode().detach()
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()\
                               .repeat(n_samples, 1, 1, 1)]
            if depth1 is not None:
                cond['depth1'] = [model.encode_first_stage((depth1.float().to(c.device))).mode().detach()]
            if mask_super is not None:
                cond["mask_cond"] = [model.encode_first_stage((mask_super.float().to(c.device))).mode().detach()]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(cond['c_concat'][0].shape[0], 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
                if depth1 is not None:
                    uc['depth1'] = [torch.zeros(cond['c_concat'][0].shape[0], 4, h // 8, w // 8).to(c.device)]
                if mask_super is not None:
                    uc["mask_cond"] = [torch.zeros(cond['c_concat'][0].shape[0], 4, h // 8, w // 8).to(c.device)]
            else:
                uc = None
            # breakpoint()
            shape = [4, h // 8, w // 8]
            mask_info = None
            if (mask_super is not None):

                samples_ddim, _, mask_info = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=cond['c_concat'][0].shape[0],
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            
            else:
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=cond['c_concat'][0].shape[0],
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

def get_depth2(pth):
    depth = np.load(pth)
    
    new_depth = np.zeros_like(depth)
    new_depth = depth.copy()
    new_depth[new_depth > 1000] = new_depth[new_depth < 1000].max() * 2
    pixels = new_depth.ravel()
    depth_min = np.percentile(pixels, 2)
    depth_max = np.percentile(pixels, 98)
    if(depth_max - depth_min < 1e-5):
        depth_min = new_depth.min()
    normalized_depth = ((new_depth - depth_min) / (depth_max - depth_min) - 0.5) * 2.0
    normalized_depth = cv2.resize(normalized_depth, (256, 256), interpolation=cv2.INTER_LINEAR)
    normalized_depth = np.tile(normalized_depth, (3, 1, 1))
    # print(normalized_depth.shape)

    depth_torch = torch.tensor(normalized_depth)
    return depth_torch

def load_mask2(path):
    mask = plt.imread(path)
    mask = np.uint8(mask * 255.)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_LINEAR)

    mask_min = mask.min()
    mask_max = mask.max()
    normalized_mask = ((mask - mask_min) / (mask_max - mask_min) - 0.5) * 2.0
    # normalized_mask = cv2.resize(normalized_mask, (256, 256), interpolation=cv2.INTER_LINEAR)
    normalized_mask = np.tile(normalized_mask, (3, 1, 1))
    return normalized_mask



device = f"cuda:0"
if not torch.cuda.is_available():
    device = "cpu"
cfg_file = 'configs/3d_mix.yaml'
config = OmegaConf.load(cfg_file)
if config.model.params.depth1:
    config.model.params.unet_config.params.in_channels += 4
if config.model.params.mask_super:
    config.model.params.unet_config.params.in_channels += 4
ckpt = 'last.ckpt'
# Instantiate all models beforehand for efficiency.
models = dict()
print('Instantiating LatentDiffusion...')
models['turncam'] = load_model_from_config(config, ckpt, device=device)
sampler = DDIMSampler(models['turncam'])

# load all the files needed
input_im = np.array(Image.open(args.input_image))
input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
input_im = input_im * 2 - 1
input_im = transforms.functional.resize(input_im, [256, 256])
mask_cond = torch.from_numpy(load_mask2(args.input_mask)).unsqueeze(0)
depth_cond = get_depth2(args.input_depth).unsqueeze(0)

# input pose
default_elevation = 15
default_dis = 1.6
default_azimuth = 0
input_RT = np.zeros((4, 4))
tmp_x = default_dis * math.cos(math.radians(default_elevation)) * math.cos(math.radians(default_azimuth))
tmp_y = default_dis * math.cos(math.radians(default_elevation)) * math.sin(math.radians(default_azimuth))
tmp_z = default_dis * math.sin(math.radians(default_elevation))
campos = np.array([tmp_x, tmp_y, tmp_z])
tar = np.array([0, 0, 0])
input_R = look_at(campos, tar)
input_RT[0:3, 0:3] = input_R
input_RT[0:3, 3] = campos
input_RT[3, 3] = 1
ref_input_RT = input_RT

# output pose
elevation = args.elevation
azimuth = args.azimuth
tmp_x = default_dis * math.cos(math.radians(elevation)) * math.cos(math.radians(azimuth))
tmp_y = default_dis * math.cos(math.radians(elevation)) * math.sin(math.radians(azimuth))
tmp_z = default_dis * math.sin(math.radians(elevation))
output_RT = np.zeros((4, 4))
campos1 = np.array([tmp_x, tmp_y, tmp_z])
tar1 = np.array([0, 0, 0])
output_R = look_at(campos1, tar1)
output_RT[0:3, 0:3] = output_R
output_RT[0:3, 3] = campos1
output_RT[3, 3] = 1
target_view_RT = (output_RT)

T = get_T2(target_view_RT, ref_input_RT).unsqueeze(0)

x_samples_ddim = sample_model(input_im, models['turncam'], sampler, 'fp32', 256, 256, 50, \
                                1, 3.0, 1.0, T, depth_cond, mask_cond)

out_img_tmp = x_samples_ddim.permute(0, 2, 3, 1)
out_img_tmp = out_img_tmp.numpy()
out_tmp = (out_img_tmp[0] * 255).astype(np.uint8)
Image.fromarray(out_tmp).save('1.png')