from PIL import Image
import numpy as np
import torchvision.transforms as Transforms
from .utils import gaussian_blur, solarize, random_color_jitter
from typing import Tuple, Union, List

class BaseAugmentationDINO:
    def __init__(self):
        self.global_1 = None
        self.global_2 = None
        self.local = None
        self.num_local_crops = None

    def __call__(self, img):
        
        if None in {self.global_1, self.global_2, self.local, self.num_local_crops}:
            nonelist = [var for var, val in {'global_1':self.global_1,\
                                             'global_2':self.global_2,\
                                             'local':self.local,\
                                             'num_local_crops':self.num_local_crops}.items() \
                        if val is None]
            print(f'{nonelist} not initialized')
            return
            
        crops = []
        crops.append(self.global_1(img))
        crops.append(self.global_2(img))
        crops.extend([self.local(img) \
                      for _ in range(self.num_local_crops)])
        return crops
# ---------------------------------------------------------------

def get_dino_augmentations(
        **kwargs
    ):
    augmentation = kwargs.get("augmentation", "AugmentationDINO").lower()
    if augmentation == "AugmentationDINO".lower():
        return AugmentationDINO(**kwargs)
    if augmentation == "AugmentationDINOSingleChannel".lower():
        return AugmentationDINOSingleChannel(**kwargs)
    elif augmentation == "AugmentationDINOexpt1".lower():
        return AugmentationDINOexpt1(**kwargs)
    elif augmentation == "AugmentationDINOexpt2".lower():
        return AugmentationDINOexpt2(**kwargs)
    else:
        raise NotImplementedError
        

def build_transforms(
            np_input: bool = True,
            crop_size: int = 64,
            scale_range = [0.0, 1.0],
            horizontal_flip_probability: float = 0.5,
            center_crop: int = 64,
            color_jitter: bool = True,
            brightness_jitter: float = 0.,
            contrast_jitter: float = 0.,
            saturation_jitter: float = 0.,
            hue_jitter: float = 0.,
            color_jitter_probability: float = 0.,
            random_grayscale: bool = True,
            grayscale_probability: float = 0.,
            random_gaussian_blur: bool = True,
            gaussian_blur_sigma: Tuple[float, float] = (0., 0.),
            gaussian_blur_probability: float = 0.,
            random_solarize: bool = True,
            solarize_probability: float = 0.,
            random_rotation: bool = True,
            rotation_degree: Union[float, Tuple[float, float]] = (0., 0.),
            normalize: bool = True,
            mean: Union[float, Tuple[float, ]] = 0.,
            std: Union[float, Tuple[float, ]] = 0.,
        ):
    transforms = []
    if np_input:
        transforms.append(Transforms.ToPILImage())
    if center_crop > 0:
        transforms.append(Transforms.CenterCrop(center_crop))
   
    transforms.extend([
            Transforms.RandomResizedCrop(crop_size,
                                 scale=scale_range,
                                 interpolation=Image.BICUBIC),
            Transforms.RandomHorizontalFlip(p=horizontal_flip_probability)
    ])
    if color_jitter:
        transforms.append(
            random_color_jitter(
                brightness_jitter = brightness_jitter,
                contrast_jitter = contrast_jitter,
                saturation_jitter = saturation_jitter,
                hue_jitter = hue_jitter,
                color_jitter_probability = color_jitter_probability,
            ))
    if random_grayscale:
        transforms.append(
            Transforms.RandomGrayscale(p=grayscale_probability)
        )
    if random_gaussian_blur:
        transforms.append(
            gaussian_blur(
                sigma = gaussian_blur_sigma,
                p=gaussian_blur_probability
            )
        )
    if random_solarize:
        transforms.append(
            solarize(p=solarize_probability)
        )
    if random_rotation:
        transforms.append(
            Transforms.RandomRotation(degrees = rotation_degree,\
                                      interpolation=Image.BICUBIC)
        )
    transforms.append(Transforms.ToTensor())
    if normalize:
        transforms.append(Transforms.Normalize(mean, std))
                        
    return Transforms.Compose(transforms)


# ---------------------------------------------------------------
class AugmentationDINO(BaseAugmentationDINO):
    '''   
    implements the standard DINO augmentations
    
    params:
        global_crop_scale_range: tuple(float, float)
            provided values must be in [0, 1]
            the stochastic global crop size is chosen as
            x*(image dimension) where x lies within the
            provided range. Two global crops are generated.
        global_crop_size: int
            resize the global crops to this size
        local_crop_scale_range: tuple(float, float)
            provided values must be 
            in [0, global_crop_scale_range[0]]
            the stochastic local crop size is chosen as
            x*(image dimension) where x lies within the
            provided range. This is smaller than global crop.
        local_crop_size: int
            resize the local crops to this size
        num_local_crops: int
            number of local crops
        dataset_mean: tuple(float, )
            tuple of length = number of channels in input image
            mean used for normalization
        dataset_std: tuple(float, )
            tuple of length = number of channels in input image
            std used for normalization
    adapted from https://github.com/facebookresearch/dino/blob/main/main_dino.py
    '''
    def __init__(self, \
            center_crop: int = 64,
            global_crop_scale_range = [0.4, 1.0],
            global_crop_size: int = 64, 
            local_crop_scale_range = [0.05, 0.4],
            local_crop_size: int = 28,
            num_local_crops: int = 8,
            dataset_mean: Tuple[float, ] = None,
            dataset_std: Tuple[float, ] = None,
            **kwargs):
        super().__init__()
        assert dataset_mean is not None
        assert dataset_std is not None
        self.global_1 = build_transforms(
                    np_input = True,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.4),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.4),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.2),
                    hue_jitter = kwargs.get('hue_jitter', 0.1),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = True,
                    grayscale_probability = kwargs.get('grayscale_probability', 0.2),
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_1', 1.0),
                    random_solarize = False,
                    random_rotation = False,
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )
        
        self.global_2 = build_transforms(
                    np_input = True,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.4),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.4),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.2),
                    hue_jitter  = kwargs.get('hue_jitter', 0.1),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = True,
                    grayscale_probability = kwargs.get('grayscale_probability', 0.2),
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_2', 0.1),
                    random_solarize = True,
                    solarize_probability = kwargs.get('solarize_probability', 0.2),
                    random_rotation = False,
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )

        self.local = build_transforms(
                    np_input = True,
                    crop_size = local_crop_size,
                    center_crop = center_crop,
                    scale_range = local_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.4),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.4),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.2),
                    hue_jitter = kwargs.get('hue_jitter', 0.1),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = True,
                    grayscale_probability = kwargs.get('grayscale_probability', 0.2),
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_local', 0.5),
                    random_solarize = False,
                    random_rotation = False,
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )

        self.num_local_crops = num_local_crops
# ---------------------------------------------------------------

# ---------------------------------------------------------------
class AugmentationDINOSingleChannel(BaseAugmentationDINO):
    '''   
    implements the standard DINO augmentations
    
    params:
        global_crop_scale_range: tuple(float, float)
            provided values must be in [0, 1]
            the stochastic global crop size is chosen as
            x*(image dimension) where x lies within the
            provided range. Two global crops are generated.
        global_crop_size: int
            resize the global crops to this size
        local_crop_scale_range: tuple(float, float)
            provided values must be 
            in [0, global_crop_scale_range[0]]
            the stochastic local crop size is chosen as
            x*(image dimension) where x lies within the
            provided range. This is smaller than global crop.
        local_crop_size: int
            resize the local crops to this size
        num_local_crops: int
            number of local crops
        dataset_mean: tuple(float, )
            tuple of length = number of channels in input image
            mean used for normalization
        dataset_std: tuple(float, )
            tuple of length = number of channels in input image
            std used for normalization
    adapted from https://github.com/facebookresearch/dino/blob/main/main_dino.py
    '''
    def __init__(self, 
            center_crop: int = 64,
            global_crop_scale_range: List[float] = [0.4, 1],
            global_crop_size: int = 64, 
            local_crop_scale_range: List[float] = [0.05, 0.4],
            local_crop_size: int = 28,
            num_local_crops: int = 8,
            dataset_mean: Tuple[float, ] = None,
            dataset_std: Tuple[float, ] = None,
            **kwargs):
        super().__init__()
        assert dataset_mean is not None
        assert dataset_std is not None

        self.global_1 = build_transforms(
                    np_input = True,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.4),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.4),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.2),
                    hue_jitter = kwargs.get('hue_jitter', 0.1),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = False,
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_1', 1.0),
                    random_solarize = False,
                    random_rotation = False,
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )
        
        self.global_2 = build_transforms(
                    np_input = True,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.4),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.4),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.2),
                    hue_jitter = kwargs.get('hue_jitter', 0.1),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = False,
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_2', 0.1),
                    random_solarize = True,
                    solarize_probability = kwargs.get('solarize_probability', 0.2),
                    random_rotation = False,
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )

        self.local = build_transforms(
                    np_input = True,
                    crop_size = local_crop_size,
                    center_crop = center_crop,
                    scale_range = local_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.4),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.4),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.2),
                    hue_jitter = kwargs.get('hue_jitter', 0.1),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = False,
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_local', 0.5),
                    random_solarize = False,
                    random_rotation = False,
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )

        self.num_local_crops = num_local_crops
# ---------------------------------------------------------------



# ---------------------------------------------------------------
class AugmentationDINOexpt1(BaseAugmentationDINO):
    '''   
    implements the standard DINO augmentations
    additionally adds random rotations 
    
    params:
        global_crop_scale_range: tuple(float, float)
            provided values must be in [0, 1]
            the stochastic global crop size is chosen as
            x*(image dimension) where x lies within the
            provided range. Two global crops are generated.
        global_crop_size: int
            resize the global crops to this size
        local_crop_scale_range: tuple(float, float)
            provided values must be 
            in [0, global_crop_scale_range[0]]
            the stochastic local crop size is chosen as
            x*(image dimension) where x lies within the
            provided range. This is smaller than global crop.
        local_crop_size: int
            resize the local crops to this size
        num_local_crops: int
            number of local crops
        dataset_mean: tuple(float, )
            tuple of length = number of channels in input image
            mean used for normalization
        dataset_std: tuple(float, )
            tuple of length = number of channels in input image
            std used for normalization
    adapted from https://github.com/facebookresearch/dino/blob/main/main_dino.py
    '''
    def __init__(self, 
            center_crop: int = 64,
            global_crop_scale_range: List[float] = [0.4, 1],
            global_crop_size: int = 64, 
            local_crop_scale_range: List[float] = [0.05, 0.4],
            local_crop_size: int = 28,
            num_local_crops: int = 8,
            dataset_mean: Tuple[float, ] = None,
            dataset_std: Tuple[float, ] = None,
            **kwargs):
        super().__init__()
        assert dataset_mean is not None
        assert dataset_std is not None

        self.global_1 = build_transforms(
                    np_input = True,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.4),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.4),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.2),
                    hue_jitter = kwargs.get('hue_jitter', 0.1),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = True,
                    grayscale_probability = kwargs.get('grayscale_probability', 0.2),
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_1', 1.0),
                    random_solarize = False,
                    random_rotation = True,
                    rotation_degree = (kwargs.get('rotation_degree', (-180., 180.))),
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )
        
        self.global_2 = build_transforms(
                    np_input = True,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.4),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.4),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.2),
                    hue_jitter = kwargs.get('hue_jitter', 0.1),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = True,
                    grayscale_probability = kwargs.get('grayscale_probability', 0.2),
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_2', 0.1),
                    random_solarize = True,
                    solarize_probability = kwargs.get('solarize_probability', 0.2),
                    random_rotation = True,
                    rotation_degree = (kwargs.get('rotation_degree', (-180., 180.))),
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )

        self.local = build_transforms(
                    np_input = True,
                    center_crop = center_crop,
                    crop_size = local_crop_size,
                    scale_range = local_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.4),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.4),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.2),
                    hue_jitter = kwargs.get('hue_jitter', 0.1),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = True,
                    grayscale_probability = kwargs.get('grayscale_probability', 0.2),
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_local', 0.5),
                    random_solarize = False,
                    random_rotation = True,
                    rotation_degree = (kwargs.get('rotation_degree', (-180., 180.))),
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )

        self.num_local_crops = num_local_crops
# ---------------------------------------------------------------

# ---------------------------------------------------------------
class AugmentationDINOexpt2(BaseAugmentationDINO):
    '''   
    implements the standard DINO augmentations
    contains augmentations that doesn't affect
    channel information
    
    params:
        global_crop_scale_range: tuple(float, float)
            provided values must be in [0, 1]
            the stochastic global crop size is chosen as
            x*(image dimension) where x lies within the
            provided range. Two global crops are generated.
        global_crop_size: int
            resize the global crops to this size
        local_crop_scale_range: tuple(float, float)
            provided values must be 
            in [0, global_crop_scale_range[0]]
            the stochastic local crop size is chosen as
            x*(image dimension) where x lies within the
            provided range. This is smaller than global crop.
        local_crop_size: int
            resize the local crops to this size
        num_local_crops: int
            number of local crops
        dataset_mean: tuple(float, )
            tuple of length = number of channels in input image
            mean used for normalization
        dataset_std: tuple(float, )
            tuple of length = number of channels in input image
            std used for normalization
    adapted from https://github.com/facebookresearch/dino/blob/main/main_dino.py
    '''
    def __init__(self, 
            center_crop: int = 64,
            global_crop_scale_range: List[float] = [0.4, 1],
            global_crop_size: int = 64, 
            local_crop_scale_range: List[float] = [0.05, 0.4],
            local_crop_size: int = 28,
            num_local_crops: int = 8,
            dataset_mean: Tuple[float, ] = None,
            dataset_std: Tuple[float, ] = None,
            **kwargs):
        super().__init__()
        assert dataset_mean is not None
        assert dataset_std is not None

        self.global_1 = build_transforms(
                    np_input = True,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = False,
                    random_grayscale = False,
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_1', 1.0),
                    random_solarize = False,
                    random_rotation = True,
                    rotation_degree = (kwargs.get('rotation_degree', (-180., 180.))),
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )
        self.global_2 = build_transforms(
                    np_input = True,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = False,
                    random_grayscale = False,
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_2', 0.1),
                    random_solarize = True,
                    solarize_probability = kwargs.get('solarize_probability', 0.2),
                    random_rotation = True,
                    rotation_degree = (kwargs.get('rotation_degree', (-180., 180.))),
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )
        self.local = build_transforms(
                    np_input = True,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = False,
                    random_grayscale = False,
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_local', 0.5),
                    random_solarize = False,
                    random_rotation = True,
                    rotation_degree = (kwargs.get('rotation_degree', (-180., 180.))),
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )
        self.num_local_crops = num_local_crops
# ---------------------------------------------------------------
        
