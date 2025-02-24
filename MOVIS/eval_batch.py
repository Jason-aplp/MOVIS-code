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
import OpenEXR, Imath
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
parser.add_argument("--config", type=str, help="path to config file")
parser.add_argument("--output_dir", type=str, help="path to output file")
parser.add_argument("--exr_res", type=int, help="specify the resolution of exr files")
parser.add_argument("--ddim_steps", type=int, default=50)
parser.add_argument("--n_samples", type=int, default=1)
parser.add_argument("--cfg_scale", type=float, default=3.0)

# Parse the arguments
args = parser.parse_args()

class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        depth1=False,
        exr_res=512,
        mask_super=False,
        validation=False,
        debug=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view
        self.depth1 = depth1
        self.mask_super = mask_super
        self.exr_res = exr_res

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]
        
        with open(os.path.join(root_dir, 'test_path.json')) as f:
            self.test_paths = json.load(f)
            # self.test_paths = self.train_paths[:40]
            
        # total_objects = len(self.paths)
        if validation:
            self.paths = self.test_paths # used last 1% as validation
        else:
            self.paths = self.train_paths # used first 99% as training
        if debug:
            self.paths = self.train_paths[:10]
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T
    
    def get_T1(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target * 1.0 / z_cond
        
        d_T = torch.tensor([d_theta.item(), d_azimuth.item(), d_z.item()])
        return d_T

    def R2E(self, R):
        sy = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

    def get_T2(self, target_RT, cond_RT):
        delta = torch.from_numpy(np.linalg.inv(cond_RT) @ target_RT).flatten()
        return delta

    def get_depth1(self, pth):
        exr_file = OpenEXR.InputFile(pth)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        depth_str = exr_file.channel('R', pt)

        depth = np.frombuffer(depth_str, dtype=np.float32)
        depth.shape = (self.exr_res, self.exr_res)
        
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

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img
    
    def load_mask1(self, path):
        mask = plt.imread(path)
        mask = np.uint8(mask * 255.)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_LINEAR)

        mask_min = mask.min()
        mask_max = mask.max()
        normalized_mask = ((mask - mask_min) / (mask_max - mask_min) - 0.5) * 2.0
        # normalized_mask = cv2.resize(normalized_mask, (256, 256), interpolation=cv2.INTER_LINEAR)
        normalized_mask = np.tile(normalized_mask, (3, 1, 1))
        return normalized_mask

    def __getitem__(self, index):

        data = {}
        total_view = self.total_view
        index_target, index_cond = random.sample(range(total_view), 2) # without replacement
        data['target'] = index_target
        data['cond'] = index_cond
        filename = os.path.join(self.root_dir, self.paths[index])
        data['scene'] = self.paths[index]
        print(data['scene'])

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]

        target_im = self.process_im(self.load_im(os.path.join(filename, '%03d_1.png' % index_target), color))
        cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d_1.png' % index_cond), color))
        target_RT = np.eye(4)
        target_RT[:3, :] = np.load(os.path.join(filename, '%03d_1.npy' % index_target))
        target_RT = np.linalg.inv(target_RT)
        cond_RT = np.eye(4)
        cond_RT[:3, :] = np.load(os.path.join(filename, '%03d_1.npy' % index_cond))
        cond_RT = np.linalg.inv(cond_RT)
        if self.depth1:
            depth_cond_pth1 = os.path.join(filename, 'depth_%03d_1.exr' % index_cond)
            depth_cond1 = self.get_depth1(depth_cond_pth1)
        if self.mask_super:
            cond_mask_pth = os.path.join(filename, 'mask_%03d' % index_cond, 'mixed_mask.png')
            target_mask_pth = os.path.join(filename, 'mask_%03d' % index_target, 'mixed_mask.png')
            cond_mask = self.load_mask1(cond_mask_pth)
            target_mask = self.load_mask1(target_mask_pth)

        data["image_target"] = target_im
        data["image_cond"] = cond_im

        if self.depth1:
            data["depth1"] = depth_cond1
        if self.mask_super:
            data["mask_cond"] = cond_mask
            data["mask_target"] = target_mask
        data["T2"] = self.get_T2(target_RT, cond_RT)
        # breakpoint()

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


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



device = f"cuda:0"
if not torch.cuda.is_available():
    device = "cpu"
cfg_file = args.config
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
trans = [torchvision.transforms.Resize(256),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))]
image_transforms = torchvision.transforms.Compose(trans)
val_dataset = ObjaverseData(root_dir=config.data.params.root_dir, image_transforms=image_transforms, exr_res = args.exr_res
                                , depth1=config.data.params.depth1,  mask_super=config.data.params.mask_super, validation=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

for i, batch in enumerate(tqdm(val_loader)):
    input_im = batch['image_cond'].to(device)
    input_im = input_im.permute(0, 3, 1, 2)
    target_im = batch['image_target'].to(device)
    target_im = target_im.permute(0, 3, 1, 2)
    T2 = batch['T2'].to(device)
    depth1_cond = batch["depth1"]
    mask_super_cond = batch["mask_cond"]
    resolution_x = 256
    resolution_y = 256
    ddim_steps = args.ddim_steps
    n_samples = args.n_samples
    cfg_scale = args.cfg_scale
    x_samples_ddim = sample_model(input_im, models['turncam'], sampler, 'fp32', resolution_x, resolution_y, ddim_steps, \
                                    n_samples, cfg_scale, 1.0, T2, depth1_cond, mask_super_cond)

    os.makedirs(os.path.join(args.output_dir, batch['scene'][0]), exist_ok=True)
    out_img_tmp = x_samples_ddim.permute(0, 2, 3, 1)
    out_img_tmp = out_img_tmp.numpy()
    out_tmp = (out_img_tmp[0] * 255).astype(np.uint8)
    Image.fromarray(out_tmp).save(os.path.join(args.output_dir, batch['scene'][0], 'pred.png'))

    input_tmp_img = input_im.detach().cpu()
    input_img_tmp = input_tmp_img.permute(0, 2, 3, 1).numpy()[0]
    input_img_tmp = (input_img_tmp - input_img_tmp.min()) / (input_img_tmp.max() - input_img_tmp.min()) * 255
    Image.fromarray(input_img_tmp.astype(np.uint8)).save(os.path.join(args.output_dir, batch['scene'][0], 'input.png'))

    target_tmp_img = target_im.detach().cpu()
    target_img_tmp = target_tmp_img.permute(0, 2, 3, 1).numpy()[0]
    target_img_tmp = (target_img_tmp - target_img_tmp.min()) / (target_img_tmp.max() - target_img_tmp.min()) * 255
    Image.fromarray(target_img_tmp.astype(np.uint8)).save(os.path.join(args.output_dir, batch['scene'][0], 'target.png'))