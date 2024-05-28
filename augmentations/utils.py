from PIL.ImageOps import solarize as Solarize
from PIL.ImageFilter import GaussianBlur 
import random
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
        return img.filter(
            GaussianBlur(
                radius=random.uniform(float(self.sigma[0]), float(self.sigma[1]))
            )
        )

class solarize:
    def __init__(
            self, \
            p: float):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return Solarize(img)

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
