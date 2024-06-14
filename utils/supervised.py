import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from models.MLP import MLP
from models import vit_backbone
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_recall_fscore_support
from typing import List
from prettytable import PrettyTable

def train_supervised(
        model,
        train_loader,
        val_loader,
        num_epochs: float, 
        optimizer, 
        scheduler,
        criterion,
    ):
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