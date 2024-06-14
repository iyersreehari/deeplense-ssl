from PIL.ImageOps import solarize as Solarize
from PIL.ImageFilter import GaussianBlur 
import numpy as np
import random
import torch
import torchvision.transforms as Transforms
from typing import Tuple

class gaussian_blur:
    def __init__(
            self,
            p: float,
            sigma: Tuple[float, float]
        ):
        self.p = p
        self.sigma = sigma 

    def __call__(
            self, 
            img):
        if random.random() > self.p:
            return img
        if isinstance(img, torch.Tensor):
            return Transforms.functional.gaussian_blur(img, 7, (float(self.sigma[0]), float(self.sigma[1])))
        else:
            return img.filter(
                GaussianBlur(
                    radius=random.uniform(float(self.sigma[0]), float(self.sigma[1]))
                )
            )

class randomrotation:
    def __init__(
            self,
            p: float,
            rotation_degree: Tuple[float, float]
        ):
        self.p = p
        self.rotation_degree = rotation_degree 

    def __call__(
            self, 
            img):
        if random.random() > self.p:
            return img
        angle = random.choice(np.arange(self.rotation_degree[0], self.rotation_degree[1]))
        if isinstance(img, torch.Tensor):
            return Transforms.functional.rotate(img, angle=angle)
        else:
            return img.rotate(angle=angle)

class solarize:
    def __init__(
            self, \
            p: float):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if isinstance(img, torch.Tensor):
            return Transforms.functional.solarize(img, img.median())
        else:
            return Solarize(img)

class MinMaxScaling:
    def __init__(
            self,
            min_clamp = -3.,
            max_clamp = 3.,
        ):
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp
        
    def __call__(self, img):
        img = torch.clamp(img, min=self.min_clamp, max=self.max_clamp)
        C, H, W = img.shape 

        max = torch.full_like(img, self.max_clamp)
        # c = torch.reshape(img, (C, -1)).max(-1).values
        # for i in range(C):
        #     max[i] = c[i]
        
        min = torch.full_like(img, self.min_clamp)
        # c = torch.reshape(img, (C, -1)).min(-1).values
        # for i in range(C):
        #     min[i] = c[i]
        
        x = max - min
        # x = x + torch.full_like(x, 1e-8)
        return (img - min) / x
        # return img

    
def random_color_jitter(
            brightness_jitter: float,
            contrast_jitter: float,
            saturation_jitter: float,
            hue_jitter: float,
            color_jitter_probability: float,
        ):
        return Transforms.RandomApply([
                    Transforms.ColorJitter(\
                        brightness=brightness_jitter,
                        contrast=contrast_jitter,
                        saturation=saturation_jitter,
                        hue=hue_jitter,)
                    ],
                    p=color_jitter_probability
                )
