"""Training Neural A* 
Author: Ryo Yonetani
Affiliation: OSX
"""
from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
import torch
from neural_astar.planner import NeuralAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt


@hydra.main(config_path="config", config_name="train")
def main(config):

    # dataloaders
    set_global_seeds(config.seed)
    train_loader = create_dataloader(
        config.dataset + ".npz", "train", config.params.batch_size, shuffle=True
    )

    # visualise the dataloaders
    print(f"train_loader : {train_loader}") 
    print(f"train_loader : {train_loader.dataset}")    
    first_item = train_loader.dataset[0]

    # print(f"Structure of first_item: {first_item}")
    print(f"Number of elements in first_item: {len(first_item)}")
    mapdesigns, startdesign, goaldesign, optimalpath = first_item
    
    if isinstance(first_item, tuple):
        print(f"Type of the first component: {type(mapdesigns)}") 
        print(f"Type of the second component: {type(startdesign)}")
    
    # visualis the feature using matplot lib
    plt.imshow(mapdesigns[0], cmap='gray')
    plt.show()


    val_loader = create_dataloader(
        config.dataset + ".npz", "valid", config.params.batch_size, shuffle=False
    )
    
    neural_astar = NeuralAstar(
        encoder_input=config.encoder.input,
        encoder_arch=config.encoder.arch,
        encoder_depth=config.encoder.depth,
        learn_obstacles=False,
        Tmax=config.Tmax,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/h_mean", save_weights_only=True, mode="max"
    )

    module = PlannerModule(neural_astar, config)
    logdir = f"{config.logdir}/{os.path.basename(config.dataset)}"
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logdir,
        max_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
