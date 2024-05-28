import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from utils import knn_accuracy
import datetime
from typing import List, Dict, Union 

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

class TrainSSL:
    def __init__(
            self,
            output_dir: str,
            expt_name: str,
            restore_from_ckpt: bool,
            restore_ckpt_path: str,
            data_path,
            data_augmentation_transforms,
            eval_transforms,
            num_classes: int,
            train_val_test_split: List[float, ],
            batch_size: int,
            num_epochs: int,
            log_freq: int,
            device: str,
            logger,
            use_mixed_precision: bool,
            #clip_grad_magnitude: float,
            **kwargs,
        ):

        self.output_dir = output_dir
        self.expt_name = expt_name
        self.expt_path = os.path.join(output_dir, f"{expt_name}_models")
        if not os.path.exists(self.expt_path):
            os.makedirs(self.expt_path)
        self.ckpt_file = os.path.join(self.expt_path, "checkpoint.pth")
        self.restore_ckpt_path = None
        if restore_from_ckpt:
            self.restore_ckpt_path= restore_ckpt_path
        self.log_freq = log_freq
        self.logger = logger 
        #self.clip_grad_magnitude = clip_grad_magnitude
        self.use_mixed_precision = use_mixed_precision
        self.student = None
        self.teacher = None
        self.num_classes = num_classes
        self.data_path = data_path
        self.batch_size = batch_size

        
        

        # Create DatasetFolder instances for each split with the respective transforms
        dataset = datasets.DatasetFolder(
            root=data_path,
            loader=npy_loader,
            extensions=['.npy'],
            transform=data_augmentation_transforms
        )
        # Define the sizes for train, validation, and test sets
        train_val_test_split = list(train_val_test_split) / np.sum(list(train_val_test_split)) 
        train_size = int(train_val_test_split[0] * len(dataset))
        val_size = int(train_val_test_split[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # Split the dataset into train, validation, and test sets
        train_indices, val_indices, test_indices = random_split(dataset, [train_size, val_size, test_size])
        
        train_dataset = datasets.DatasetFolder(
            root=data_path,
            loader=npy_loader,
            extensions=['.npy'],
            transform=eval_transforms
        )

        val_dataset = datasets.DatasetFolder(
            root=data_path,
            loader=npy_loader,
            extensions=['.npy'],
            transform=eval_transforms
        )
        
        test_dataset = datasets.DatasetFolder(
            root=data_path,
            loader=npy_loader,
            extensions=['.npy'],
            transform=eval_transforms
        )

        # Apply the indices to get the correct subsets
        dataset.samples = [dataset.samples[i] for i in train_indices.indices]
        train_dataset.samples = [train_dataset.samples[i] for i in train_indices.indices]
        val_dataset.samples = [val_dataset.samples[i] for i in val_indices.indices]
        test_dataset.samples = [test_dataset.samples[i] for i in test_indices.indices]


        self.dataloader = DataLoader(
            dataset,
            batch_size = self.batch_size,
            num_workers = kwargs.get('num_workers', 4),
            pin_memory = kwargs.get('pin_memory', True),
            drop_last = kwargs.get('drop_last', True),
        )
        
        self.eval_train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size = self.batch_size,
            num_workers = kwargs.get('num_workers', 4),
            pin_memory = kwargs.get('pin_memory', True),
            drop_last = kwargs.get('drop_last', True),
        )
        self.eval_val_dataloader = None
        if val_size!=0:
            self.eval_val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size = self.batch_size,
                num_workers = kwargs.get('num_workers', 4),
                pin_memory = kwargs.get('pin_memory', True),
                drop_last = kwargs.get('drop_last', True),
            ) 
            
        self.eval_test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size = self.batch_size,
            num_workers = kwargs.get('num_workers', 4),
            pin_memory = kwargs.get('pin_memory', True),
            drop_last = kwargs.get('drop_last', True),
        )
        self.logger.info(f"Loaded Dataset from {self.data_path}. Dataset contains {len(dataset)} images")

        self.epochs = num_epochs
        self.steps_per_epoch = len(self.dataloader)
        
        self.optimizer = None
        self.criterion = None
        
        self.device = device           
        
        self.history = {
            "loss_stepwise": [],
            "loss_epochwise": [],
            "knn_top1": [],
            "knn_top5": [],
        }
        self.state = None

    def _init_state(self):
        self.state = {
            "info": {
                "output_dir": self.output_dir,
                "expt_name": self.expt_name,
                "ckpt_file": self.ckpt_file,
                "self.device": self.device,
            }, 
            "student": self.student,
            "teacher": self.teacher,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "history": self.history,
            "current_epoch": 0
         }
        self.fp16_scaler = None
        if self.use_mixed_precision:
            self.fp16_scaler = torch.cuda.amp.GradScaler()
            self.state["fp16_scaler"]: self.fp16_scaler

        if self.restore_ckpt_path is not None:
            self._restore()

    def _restore(self) -> None:
        assert self.restore_ckpt_path is not None, "Checkpoint path to restore not provided"
        
        ckpt_dict = torch.load(self.restore_ckpt_path, map_location="cpu")

        for restore_item in self.state:
            if restore_item == "info":
                continue
            elif restore_item in ["student", "teacher", "fp16_scaler"]:
                msg = self.state[restore_item].load_state_dict(ckpt_dict[restore_item].state_dict(), strict=False)
            else:
                msg = self.state[restore_item].load_state_dict(ckpt_dict[restore_item], strict=False)
            self.logger.info(f"Loaded {restore_item} from checkpoint {self.restore_ckpt_name}\n\t{msg}")

        self.logger.info(f"Restored checkpoint from {self.restore_ckpt_name}")
                
    def update_history(self, step: Dict)->None:
        for key in self.history:
            if step[key] is not None:
                self.history[key].append(step[key])

    def save_checkpoint(self, ckpt_file):
        torch.save(self.state, ckpt_file)

    def forward_teacher(self, img):
        return self.teacher(img) 

    def forward_student(self, img):
        return self.student(img)

    def update_teacher(self):
        raise NotImplementedError

    def process_grads_before_step(self, epoch): 
        return

    def get_lr_and_wd(self, current_step):
        if self.lr is None or self.wd is None:
            raise NotImplementedError
        return self.lr, self.wd

    @torch.no_grad()
    def update_teacher(self):
        raise NotImplementedError
        
    def train_one_epoch(self):
        losses = []
        for idx, (img, _) in enumerate(self.dataloader):

            cur_step = self.state["current_epoch"]*self.steps_per_epoch + idx
            lr, wd = self.get_lr_and_wd(cur_step)
            
            for param_idx, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = lr
                if param_idx == 0:  
                    param_group["weight_decay"] = wd
            img = [im.to(self.device, non_blocking=True) for im in img]
            
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                teacher_output = self.forward_teacher(img)  
                student_output = self.forward_student(img)
                loss = self.criterion(student_output, teacher_output, self.state["current_epoch"])
    
            if np.isinf(loss.item()):
                self.logger.error(f"Loss is Infinite. Training stopped")
                sys.exit(1)

            if self.fp16_scaler is None:
                loss.backward()
                self.process_grads_before_step(self.state["current_epoch"])
                self.optimizer.step()
                
            else:
                self.fp16_scaler.scale(loss).backward()
                self.process_grads_before_step(self.state["current_epoch"])
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()

            self.update_teacher(cur_step)
            losses.append(loss.item())
            
        self.history["loss_stepwise"].extend(losses)
        self.history["loss_epochwise"].append(np.mean(losses))
        return np.mean(losses)
    
    def train(self):
        if self.state is None:
            self._init_state()
        for epoch in range(self.state["current_epoch"], self.epochs):
            start_time = time.time()
            loss = self.train_one_epoch()
            total_time = time.time() - start_time
            self.logger.info(f"Epoch: {epoch} finished in {datetime.timedelta(seconds=int(total_time))} seconds,")
            self.logger.info(f"\tLoss: {loss:.6e}")
            knn_top1, knn_top5 = None, None
            if self.state["current_epoch"]%self.log_freq == 0 or self.state["current_epoch"] == self.epochs:
                knn_top1, knn_top5 = self.compute_knn_accuracy()
                #self.logger.info(f"\tKNN: {knn_accuracy:.6e}")
            self.history["knn_top1"].append(knn_top1)
            self.history["knn_top5"].append(knn_top5)
            self.save_checkpoint(self.ckpt_file)
            if self.state["current_epoch"]%self.log_freq == 0 or self.state["current_epoch"] == self.epochs:
                self.save_checkpoint(os.path.join(self.expt_path, f"epoch_{self.state['current_epoch']}_accknn_{knn_top1:.6f}_checkpoint.pth"))
            self.state["current_epoch"] += 1

    @torch.no_grad()
    def compute_knn_accuracy(
            self, 
            mode: str = None,
            knn_k: int=20,
        ):
        self.logger.info("Testing accuracy of learned features using KNN")
        test_dataloader = None
        if mode is None:
            if self.eval_val_dataloader is not None:
                mode = "Val"
            else:
                mode = "Test"
        if mode == "Val":
            test_dataloader=self.eval_val_dataloader
        elif mode == "Test":
            test_dataloader=self.eval_test_dataloader
        else:
            self.logger.error(f"mode {mode} not defined")
        top1, top5 = knn_accuracy(
            model=self.student.backbone,
            train_dataloader=self.eval_train_dataloader,
            test_dataloader=test_dataloader,
            classes=self.num_classes, 
            knn_k=knn_k,
            device=self.device
        )
        self.logger.info(f"\t{mode}: KNN Acc@1:{top1:.6f}, KNN Acc@5:{top5:.6f} with neighbour count: {knn_k}")
        return top1, top5
    
    
        
        

        
