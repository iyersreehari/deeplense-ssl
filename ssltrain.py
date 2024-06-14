import os
import sys
import logging
from yaml import safe_load, safe_dump
import numpy as np
import torch
import torchvision.transforms as Transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from utils import get_system_info, set_seed
from models import vit_backbone
from ssltraining.dino import TrainDINO
from models import vit_backbone
from datetime import datetime
from augmentations import get_dino_augmentations
from augmentations.utils import MinMaxScaling


# utility function to update config yaml from default
def update_dict(args, config_args):
    for key in config_args:
        if key in args:
            if isinstance(args[key], dict):
                update_dict(args[key], config_args[key])
            else:
                args[key] = config_args[key]
        else:
            args[key] = config_args[key]

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def main():
    # Ensure a config file is provided as a command-line argument
    if len(sys.argv) not in [2,3]:
        print("Usage: python main.py <config_file> <optional_default_config_file>")
        sys.exit(1)


    config_file = sys.argv[1]
    # dict with default args
    args = None
    if len(sys.argv) == 2:
        args = safe_load(open(os.path.join(os.curdir, *["configs", "defaults.yaml"]), "r"))
    else:
        args = safe_load(open(sys.argv[2], "r"))
    # will be updated from the parsed config yaml file
    config_args = safe_load(open(config_file, "r"))
    update_dict(args, config_args)

    args["experiment"]["output_dir"] = f"{args['experiment']['output_dir']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not os.path.exists(args["experiment"]["output_dir"]):
        os.makedirs(args["experiment"]["output_dir"])

    set_seed(args["experiment"]["seed"], args["experiment"]["device"])

    # Retrieve system information
    system_info = get_system_info()
    safe_dump(system_info, open(os.path.join(args["experiment"]["output_dir"], "sysinfo.yaml"), "w"))

    # create logger
    log_file = os.path.join(args["experiment"]["output_dir"], 'logs.txt')
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        force=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info(f"logger initialized")

    # backbone for ssl
    student_backbone = None
    teacher_backbone = None
    backbone = args["network"]["backbone"]
    if backbone.startswith("vit"):
        arch_args = [tuple(args["input"]["image size"]), args["input"]["channels"], args["network"]["patch_size"]]
        student_backbone = vit_backbone(backbone, *arch_args)
        teacher_backbone = vit_backbone(backbone, *arch_args)
    else:
        print(f"`backbone` specified as {backbone} which is not implemented. Exiting.")
        sys.exit(1)

    logger.info(f"student and teacher networks initialized")

    # load data and initialize transforms
    assert args["input"]["data path"] is not None, "Input data path cannot be `None`"
    transform = Transforms.Compose([
        #Transforms.ToPILImage(),
        # Transforms.ToTensor(),
        Transforms.CenterCrop(args["ssl augmentation kwargs"]["center_crop"]),
        MinMaxScaling()
    ])
    loader = DataLoader(
        dataset=datasets.DatasetFolder(root=args["input"]["data path"], 
                                        loader=npy_loader, 
                                        extensions=['.npy'],
                                        transform=transform),
        batch_size=64,
        num_workers=1,
        shuffle=False
    )
    
    # Initialize mean and std 
    mean = torch.zeros(3) if args["input"]["channels"] == 3 else torch.zeros(1)
    std = torch.zeros(3) if args["input"]["channels"] == 3 else torch.zeros(1)
    maximum = 0
    nb_samples = 0
    dtype = None
    # Iterate over the dataset
    for data, _ in loader:
        if len(data.shape) == 3:
            data = data.unsqueeze(1)
        batch_samples = data.size(0)  # batch size (the number of images in the current batch)
        data = data.view(batch_samples, data.size(1), -1)  # reshape to (batch_size, channels, H*W)
        mean += data.mean(-1).sum(0)  # sum the means over all pixels in each channel
        std += data.std(-1).sum(0)    # sum the standard deviations over all pixels in each channel
        maximum = max(maximum, data.max()) 
        dtype = str(data.dtype)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    args["ssl augmentation kwargs"]["dataset_mean"] = mean.tolist()
    args["ssl augmentation kwargs"]["dataset_std"] = std.tolist()

    logger.info(f"Dataset mean: {args['ssl augmentation kwargs']['dataset_mean']}\nDataset std: {args['ssl augmentation kwargs']['dataset_std']}, {maximum}, {dtype}")

    assert torch.all(torch.isfinite(mean)), "Mean is not finite"
    assert torch.all(torch.isfinite(std)), "Std is not finite"
    
    
    data_augmentation_transforms = None
    if args["experiment"]["ssl_training"].lower() == "dino":
        data_augmentation_transforms = get_dino_augmentations(**args["ssl augmentation kwargs"]) 
    eval_transforms = Transforms.Compose([
        #Transforms.ToPILImage(),
        # Transforms.ToTensor(),
        Transforms.CenterCrop(args["ssl augmentation kwargs"]["center_crop"]),
        MinMaxScaling(),
        #Transforms.ToTensor(),
        Transforms.Normalize(args["ssl augmentation kwargs"]["dataset_mean"], \
                             args["ssl augmentation kwargs"]["dataset_std"]),
    ])

     
    # assert args["train args"]["batch_size"] % args["train args"]["physical_batch_size"] == 0
    
    # initialize ssl training object
    ssl_training = None
    if args["experiment"]["ssl_training"] is None:
        print("`ssl_training` which specifies the training method cannot be `None`. Exiting.")
        sys.exit(1)
    elif args["experiment"]["ssl_training"].lower() == "dino":
        ssl_training = TrainDINO(
            output_dir = args["experiment"]["output_dir"],
            expt_name = args["experiment"]["expt_name"],
            logger = logger,
            student_backbone = student_backbone,
            teacher_backbone = teacher_backbone,
            data_path = args["input"]["data path"],
            data_augmentation_transforms = data_augmentation_transforms,
            eval_transforms = eval_transforms,
            num_classes = args["input"]["num classes"],
            train_val_test_split = tuple(args["train args"]["train_val_test_split"]),
            batch_size = args["train args"]["batch_size"],
            #physical_batch_size = args["train args"]["physical_batch_size"],
            embed_dim = student_backbone.embed_dim,
            num_local_crops = args["ssl augmentation kwargs"]["num_local_crops"],
            scheduler_warmup_epochs = args["optimizer"]["scheduler_warmup_epochs"],
            warmup_teacher_temp = args["optimizer"]["warmup_teacher_temp"],
            teacher_temp = args["optimizer"]["teacher_temp"],
            warmup_teacher_temp_epochs = args["optimizer"]["warmup_teacher_temp_epochs"],
            momentum_teacher = args["optimizer"]["momentum_teacher"],
            num_epochs = args["train args"]["num_epochs"],
            head_output_dim = args["network"]["head_output_dim"],
            head_hidden_dim = args["network"]["head_hidden_dim"],
            head_bottleneck_dim = args["network"]["head_bottleneck_dim"],
            restore_from_ckpt = args["restore"]["restore"],
            restore_ckpt_path = args["restore"]["ckpt_path"],
            lr = args["optimizer"]["init_lr"],
            final_lr = args["optimizer"]["final_lr"],
            weight_decay = args["optimizer"]["init_wd"],
            final_weight_decay = args["optimizer"]["final_wd"],
            clip_grad_magnitude = args["optimizer"]["clip_grad_magnitude"],
            head_use_bn = args["network"]["head_use_bn"],
            head_norm_last_layer = args["network"]["head_norm_last_layer"],
            head_nlayers = args["network"]["head_nlayers"],
            optimizer = args["optimizer"]["optimizer"],
            log_freq = args["experiment"]["log_freq"],
            device = args["experiment"]["device"],
            use_mixed_precision = args["experiment"]["use_mixed_precision"],
            freeze_last_layer = args["train args"]["freeze_last_layer"],
            knn_neighbours = args["train args"]["knn_neighbours"]
        )
    else:
        print(f"Specified `ssl_training` method {args['experiment']['ssl_training']} not implemented. Exiting.")
        sys.exit(0)

    # train
    ssl_training.train()

    # is redundant
    # save student backbone model
    torch.save(ssl_training.student.backbone, os.path.join(args["experiment"]["output_dir"], 'representation_network.pth'))
    


if __name__ == "__main__":
    main()

    
    
    
    
    
