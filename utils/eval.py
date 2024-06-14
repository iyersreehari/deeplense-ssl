import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as Transforms
from models.MLP import MLP
from torchvision import datasets
from torch.utils.data import DataLoader
from augmentations.utils import MinMaxScaling
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_recall_fscore_support
from typing import List
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay
import copy
import matplotlib.pyplot as plt

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def get_dataloader(
        data_path: str,
        eval_transforms: Transforms,
        indices: List[int],
        batch_size: int,
        shuffle: bool,
    ):
    dataset = datasets.DatasetFolder(
        root=data_path,
        loader=npy_loader,
        extensions=['.npy'],
        transform=eval_transforms
    )
    if indices is not None:
        dataset.samples = [dataset.samples[i] for i in indices.indices]
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class GenerateParams:
    def __init__(
            self,
            num_epochs: List[float], 
            lr: List[float],
            momentum: List[float],
            weight_decay: List[float],
        ):
        self.param_grid = np.array(np.meshgrid(lr, momentum, weight_decay, num_epochs)).T.reshape(-1,4)
    def next(self):
        for i in range(len(self.param_grid)):
            param = {}
            param["lr"] = self.param_grid[i][0]
            param["momentum"] = self.param_grid[i][1]
            param["weight_decay"] = self.param_grid[i][2]
            param["num_epochs"] = self.param_grid[i][3]
            yield param
            
def train(
        model,
        ckpt_path,
        eval_transforms,
        num_epochs: float, 
        optimizer, 
        scheduler,
        criterion,
        data_path,
        batch_size = 32,
    ):
    state = torch.load(ckpt_path, map_location="cpu")
    
    # regenerate the datasets
    train_indices = state["history"]["train_indices"]
    val_indices = state["history"]["val_indices"]

    train_loader = get_dataloader(data_path, eval_transforms, 
                    train_indices, batch_size=batch_size, 
                    shuffle=True)
    val_loader = get_dataloader(data_path, eval_transforms, 
                    val_indices, batch_size=batch_size, 
                    shuffle=False)
    best_loss, best_val_loss, best_acc, best_model = None, None, None, None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.flatten(), labels.flatten().to(torch.float))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs.flatten(), labels.flatten().to(torch.float))
                val_loss += loss.item()
                predicted = (outputs.cpu().numpy().ravel() > 0).astype(np.float16)
                # val_preds.extend(predicted.tolist())
                # val_y.extend(labels.tolist())
                total += labels.size(0)
                correct += (np.count_nonzero(predicted == labels.cpu().numpy().ravel().astype(np.float16)))
        val_loss = val_loss/len(val_loader)
        acc = 100 * correct / total
        if best_acc is None or best_acc < acc:
            best_val_loss = val_loss
            best_loss = running_loss/len(train_loader)
            best_model = copy.deepcopy(model)
            best_acc = acc
        print(f'[{epoch+1}/{num_epochs}] Train Loss: {best_loss:.4f}, '
              f'Val Loss: {best_val_loss:.4f}, '
              f'Val Accuracy: {best_acc:.2f}%\n')
    return best_model
        
def _hyperparam_grid(
        ckpt_path, 
        eval_transforms,
        embed_dim: int,
        num_epochs: List[float], 
        lr: List[float],
        momentum: List[float],
        weight_decay: List[float],   
        data_path: str = "../input/real_lenses_dataset",
        net = None,
        mode = "finetune"
    ):
    
    state = torch.load(ckpt_path, map_location="cpu")

    params = GenerateParams(
        num_epochs, 
        lr,
        momentum,
        weight_decay
    )
    best_loss = None
    best_params = None
    for iter, param in enumerate(params.next()):
        print("="*96)
        lr = param["lr"]
        momentum = param["momentum"]
        weight_decay = param["weight_decay"]
        num_epochs = int(param["num_epochs"])
        print(f"Iter {iter+1}")
        print(f"lr: {lr}, momentum: {momentum}, weight_decay: {weight_decay}, num_epochs: {num_epochs}\n")
        model = None
        if net is not None:
            model = copy.deepcopy(net)
        elif mode == "finetune":
            model = MLP(
                copy.deepcopy(state["student"].backbone),
                embed_dim = embed_dim,
                output_dim = 1,
                freeze_backbone = False,
            )
        else:
            model = MLP(
                copy.deepcopy(state["student"].backbone),
                embed_dim = embed_dim,
                output_dim = 1,
                freeze_backbone = True,
            )
        torch.nn.init.xavier_uniform_(model.fc.weight)
        model.fc.bias.data.fill_(0.01)
        
        model = model.cuda()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        model = train(model, ckpt_path, eval_transforms, num_epochs, optimizer, scheduler, criterion, data_path)
        loss, acc = test_model(model, ckpt_path, eval_transforms, criterion, data_path)
        if best_loss is None:
            best_loss = loss
            best_params = param
        elif best_loss > loss:
            best_loss = loss
            best_params = param
        else:
            continue
        print("="*96)
        print("\n")
        
    return best_params

def hyperparam_grid_finetune(
        ckpt_path, 
        eval_transforms,
        embed_dim: int,
        num_epochs: List[float], 
        lr: List[float],
        momentum: List[float],
        weight_decay: List[float],   
        data_path: str = "../input/real_lenses_dataset",
    ):
    return _hyperparam_grid(
                    ckpt_path = ckpt_path, 
                    eval_transforms = eval_transforms,
                    embed_dim = embed_dim,
                    num_epochs = num_epochs, 
                    lr = lr,
                    momentum = momentum,
                    weight_decay = weight_decay,   
                    data_path = data_path,
                    mode = "finetune"
                )

def hyperparam_grid_lp(
        ckpt_path, 
        eval_transforms,
        embed_dim: int,
        num_epochs: List[float], 
        lr: List[float],
        momentum: List[float],
        weight_decay: List[float],   
        data_path: str = "../input/real_lenses_dataset",
    ):
    return _hyperparam_grid(
                    ckpt_path = ckpt_path, 
                    eval_transforms = eval_transforms,
                    embed_dim = embed_dim,
                    num_epochs = num_epochs, 
                    lr = lr,
                    momentum = momentum,
                    weight_decay = weight_decay,   
                    data_path = data_path,
                    mode = "lp"
                )

def test_model(
        model,
        ckpt_path, 
        eval_transforms,
        criterion,
        data_path: str = "../input/real_lenses_dataset",
    ):    
    state = torch.load(ckpt_path, map_location="cpu")
    test_indices = state["history"]["test_indices"]

    test_loader = get_dataloader(data_path, eval_transforms, 
                    test_indices, batch_size=32, 
                    shuffle=False)
    val_preds = []
    val_y = []
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    test_loss = 0
    output = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        output.extend(outputs.tolist())
        loss = criterion(outputs.flatten(), labels.flatten().to(torch.float))
        test_loss += loss.item()
        val_y.extend(labels.tolist())
    y_true=np.array(val_y).ravel().astype(np.float16)
    y_pred=(np.array(output).ravel() > 0).astype(np.float16)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Evaluation on held out test dataset")
    print(f"Confusion Matrix")
    t = PrettyTable(['', 'predicted lenses', 'predicted nonlenses'])
    t.add_row(['true lenses', cm[0][0], cm[0][1]])
    t.add_row(['true nonlenses', cm[1][0], cm[1][1]])
    print(t) 
    
    loss /= len(test_loader)
    acc = accuracy_score(y_true, y_pred)*100
    
    print("Test Metrics")
    t = PrettyTable(header=False)
    t.add_row(['accuracy', f'{acc:.4f}%'])
    t.add_row(['loss', f'{loss:.4f}'])
    auc = roc_auc_score(y_true, np.array(output).ravel())
    t.add_row(['auc score', f'{auc:.4f}'])
    print(t)
    
    t = PrettyTable(['', 'precision', 'recall', 'f-score', 'support'])
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0.0)
    t.add_row(['lenses', f'{precision[0]:.4f}', f'{recall[0]:.4f}', f'{f_score[0]:.4f}', f'{support[0]}'])
    t.add_row(['nonlenses', f'{precision[1]:.4f}', f'{recall[1]:.4f}', f'{f_score[1]:.4f}', f'{support[1]}'])
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0.0)
    t.add_row(['macro averaged', f'{precision:.4f}', f'{recall:.4f}', f'{f_score:.4f}', ''])
    print(t)

    return output, y_true, acc, auc

def plot_cm_roc(y_score, y):
    y_score = np.array(y_score)
    y = np.array(y)

    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    cm = confusion_matrix(y, y_score.ravel() > 0)
    ConfusionMatrixDisplay(cm).plot(ax=ax[0])
    ax[0].set_title('Confusion Matrix')
    
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f'Class Lenses').plot(ax=ax[1])
    
    ax[1].set_title('ROC AUC Curves')
    plt.tight_layout()
    plt.show()