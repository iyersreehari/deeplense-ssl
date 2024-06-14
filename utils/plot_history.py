import torch
import os
import matplotlib.pyplot as plt
import numpy as np



def plot_history(ckpt_path: str, plt_save_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    loss = state['history']['loss_epochwise']
    loss_stepwise = state['history']['loss_stepwise']
    steps_per_epoch = len(loss_stepwise) / len(loss)
    knntop1 = [top1 for top1 in state['history']['knn_top1'] if top1 is not None]
    
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))
    axs = axs.flatten()
    
    x = np.arange(0, len(loss))
    axs[0].plot(x, loss)
    axs[0].grid(alpha = 0.5)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss vs Epoch")
    
    x = np.arange(0, len(knntop1)) * 50
    axs[1].plot(x, knntop1)
    axs[1].grid(alpha = 0.5)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Val KNN Top@1 Accuracy")
    axs[1].set_title("Val KNN Top@1 Accuracy vs Epoch")
    # axs[1].set_xticks(np.linspace(0, len(knntop1), 15, dtype=np.int16).tolist()*10)
    
    plt.show()
    plt.savefig(plt_save_path)


