import os

import hydra
import moviepy.editor as mpy
from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader, visualize_results
from neural_astar.utils.training import load_from_ptl_checkpoint

import torch 
from torchvision.utils import save_image
import numpy as np


@hydra.main(config_path="config", config_name="create_gif")
def main(config):
    dataname = os.path.basename(config.dataset)

    if config.planner == "na":
        planner = NeuralAstar(
            encoder_input=config.encoder.input,
            encoder_arch=config.encoder.arch,
            encoder_depth=config.encoder.depth,
            learn_obstacles=False,
            Tmax=1.0,
        )
        planner.load_state_dict(
            load_from_ptl_checkpoint(f"{config.modeldir}/{dataname}")
        )

    else:
        planner = VanillaAstar()

    problem_id = config.problem_id
    savedir = f"{config.resultdir}/{config.planner}"
    os.makedirs(savedir, exist_ok=True)

    dataloader = create_dataloader(
        config.dataset + ".npz",
        "test",
        100,
        shuffle=False,
        num_starts=1,
    )
    map_designs, start_maps, goal_maps, opt_trajs = next(iter(dataloader))
    outputs = planner(
        map_designs, start_maps, goal_maps, store_intermediate_results=True
    )

    outputs = planner(
        map_designs[problem_id : problem_id + 1],
        start_maps[problem_id : problem_id + 1],
        goal_maps[problem_id : problem_id + 1],
        store_intermediate_results=True,
    )
    frames = []
    # log map_designs here
    # print(f" map designs : {map_designs[problem_id : problem_id + 1].shape}")

    for intermediate_results in outputs.intermediate_results:
        # intermediate_results["map"] = map_designs[problem_id : problem_id + 1]
        # print(f"inter : {intermediate_results}")
        frame = visualize_results(
            map_designs[problem_id : problem_id + 1], intermediate_results, scale=4
        )
        # print(viz)
        frames.append(frame)

    
    # frames = [
    #     visualize_results(
    #         map_designs[problem_id : problem_id + 1], intermediate_results, scale=4
    #     )
    #     for intermediate_results in outputs.intermediate_results
    # ]
    # save frames as .jpg
    # t = savedir
    # print(f"t : {t}")
    # exit()
    for i, frame in enumerate(frames):
        frame_tensor = torch.from_numpy(frame).permute(2,0,1).float() / 255.0
        save_image(frame_tensor, f"{savedir}/frame_{i:04d}.jpg")
        # frame is ndarray, save it as jpg.  frame is a 3D array  (H, W, C)   C is 3     (RGB)  0-255  uint8 
        # convert to image and save it.
        #              
        # frame.save(f"{savedir}/frame_{i:04d}.jpg")

    clip = mpy.ImageSequenceClip(frames + [frames[-1]] * 15, fps=30)
    clip.write_gif(f"{savedir}/video_{dataname}_{problem_id:04d}.gif")
    clip.write_videofile(f"{savedir}/video_{dataname}_{problem_id:04d}.mp4")


if __name__ == "__main__":
    main()
