import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .base import TrainSSL
from models import DINOHead, MultiCropWrapper
from losses import DINOLoss
from utils import cosine_scheduler
from typing import List, Dict, Union, Optional, Tuple

class TrainDINO(TrainSSL):
    def __init__(
            self,
            output_dir: str,
            expt_name: str,
            logger,
            student_backbone: nn.Module,
            teacher_backbone: nn.Module,
            data_path,
            data_augmentation_transforms,
            eval_transforms,
            num_classes: int,
            train_val_test_split: Tuple[float, ],
            batch_size: int,
            embed_dim: int,
            num_local_crops: int,
            scheduler_warmup_epochs: int,
            warmup_teacher_temp: float,
            teacher_temp: float,
            warmup_teacher_temp_epochs: int,
            momentum_teacher: float,
            num_epochs: int,
            head_output_dim: int,
            head_hidden_dim: int,
            head_bottleneck_dim: int,
            clip_grad_magnitude: float = 0.3,
            restore_from_ckpt: bool = False,
            restore_ckpt_path: Optional[str] = None,
            lr: float = 1e-3,
            final_lr: Optional[float] = None,
            weight_decay: float = 5e-3,
            final_weight_decay: Optional[float] = None,
            head_use_bn: bool = False,
            head_norm_last_layer: bool = True,
            head_nlayers: int = 3,
            optimizer: Union["AdamW, Adam, SGD, LARS"] = "AdamW",
            log_freq: int = 5,
            device: str = "cuda",
            use_mixed_precision: bool = "True",
            freeze_last_layer: int = 1,
            **kwargs,
            
        ):
        
        super().__init__(
            output_dir=output_dir,
            expt_name=expt_name,
            restore_from_ckpt=restore_from_ckpt,
            restore_ckpt_path=restore_ckpt_path,
            data_path=data_path,
            data_augmentation_transforms=data_augmentation_transforms,
            eval_transforms=eval_transforms,
            num_classes=num_classes,
            train_val_test_split = train_val_test_split,
            batch_size=batch_size,
            num_epochs = num_epochs,
            log_freq = log_freq,
            device=device,
            logger=logger,
            use_mixed_precision=use_mixed_precision,
            **kwargs
        )
        self.logger.info("Initializing DINO training")

        self.num_local_crops = num_local_crops
        self.clip_grad_magnitude = clip_grad_magnitude
        self.freeze_last_layer = freeze_last_layer
        #lr = min_lr if min_lr is not None else lr
        #min_weight_decay = min_weight_decay if min_weight_decay is not None else weight_decay
        

        # student and teacher network
        self.student = MultiCropWrapper(
                    backbone = student_backbone,
                    head = DINOHead(
                        input_dim = embed_dim, 
                        output_dim = head_output_dim, 
                        hidden_dim = head_hidden_dim,
                        bottleneck_dim = head_bottleneck_dim,
                        use_bn = head_use_bn, 
                        norm_last_layer = head_norm_last_layer, 
                        nlayers = head_nlayers
                    ),
                )
        self.teacher = MultiCropWrapper(
                    backbone = teacher_backbone,
                    head = DINOHead(
                        input_dim = embed_dim, 
                        output_dim = head_output_dim, 
                        hidden_dim = head_hidden_dim,
                        bottleneck_dim = head_bottleneck_dim,
                        use_bn = head_use_bn, 
                        nlayers = head_nlayers
                    ),
                )
        self.student, self.teacher = self.student.to(device), self.teacher.to(device)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.load_state_dict(self.student.state_dict())

        self.logger.info("Built Student and Teacher networks.\nBoth are initialized with same parameters")

        # loss
        self.criterion = DINOLoss(
                            output_dim = head_output_dim,
                            num_crops = 2 + self.num_local_crops,
                            warmup_teacher_temp = warmup_teacher_temp,
                            teacher_temp = teacher_temp,
                            warmup_teacher_temp_epochs = warmup_teacher_temp_epochs,
                            nepochs = self.epochs
                        ).to(self.device)

        # optimizer
        # lr is assigned by scheduler
        try:
            if optimizer == "AdamW":
                self.optimizer = torch.optim.AdamW(self.student.parameters())
            elif optimizer == "Adam":
                self.optimizer = torch.optim.Adam(self.student.parameters()) 
            elif optimizer == "SGD":
                self.optimizer = torch.optim.SGD(self.student.parameters(), lr=0., momentum=0.9) 
            elif optimizer == "LARS":
                raise NotImplementedError(f"{optimizer} not implemented")
            else:
                raise NotImplementedError(f"{optimizer} not implemented")
        except NotImplementedError as error:
            self.logger.error(error)
            raise error

        self.logger.info(f"Initialized {optimizer} Optimizer")

        # scheduler
        self.lr_schedule = cosine_scheduler(
            init_val = lr * batch_size / 256., 
            final_val = final_lr,
            epochs = self.epochs, 
            steps_per_epoch = self.steps_per_epoch,
            warmup_epochs = scheduler_warmup_epochs,
        )
        self.wd_schedule = cosine_scheduler(
            init_val = weight_decay,
            final_val = final_weight_decay,
            epochs = self.epochs, 
            steps_per_epoch = self.steps_per_epoch,
        )
        self.momentum_schedule = cosine_scheduler(
            init_val = momentum_teacher, 
            final_val = 1,
            epochs = self.epochs, 
            steps_per_epoch = self.steps_per_epoch,
        )
        self.logger.info(f"Initialized schedulers")

        self._init_state()

    def forward_teacher(self, img):
        return self.teacher(img[:2]) 

    def forward_student(self, img):
        return self.student(img)

    def get_lr_and_wd(self, current_step):
        return self.lr_schedule[current_step], self.wd_schedule[current_step]


    def process_grads_before_step(self, epoch): 
        if self.clip_grad_magnitude != 0:
            if self.fp16_scaler is not None:
                self.fp16_scaler.unscale_(self.optimizer)
            for param in self.student.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    clip_coef = self.clip_grad_magnitude / (param_norm + 1e-6)
                    if clip_coef < 1:
                        param.grad.data.mul_(clip_coef)
        if self.freeze_last_layer != 0 and epoch < self.freeze_last_layer:
            self._cancel_gradients_last_layer(epoch, self.student, self.freeze_last_layer)
        return
      

    def _cancel_gradients_last_layer(self, epoch, model, freeze_last_layer):
        if epoch >= freeze_last_layer:
            return
        for n, p in model.named_parameters():
            if "last_layer" in n:
                p.grad = None
        return

    @torch.no_grad()
    def update_teacher(self, current_step):
        teacher_momentum = self.momentum_schedule[current_step]  # momentum parameter
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.mul_(teacher_momentum)
            teacher_param.data.add_((1 - teacher_momentum) * student_param.detach().data)
    

        






        
