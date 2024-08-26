import numpy as np
import os
import torch
import imageio
import matplotlib.pyplot as plt
from PIL import Image

path = path =  './planning-datasets/data/mpd/mazes_032_moore_c8.npz'
output_path = './test1.npy'

with np.load(path) as data:
    # map_designs present in the dataset and choosing a single map_design
    map_designs = data['arr_0'].astype(np.float32)
    map_design = map_designs[0][np.newaxis]

    # goal_maps present in the dataset and choosing a single goal_map
    goal_maps = data['arr_1'].astype(np.float32)  
    goal_map = goal_maps[0]

    # opt_policies present in the dataset and choosing a single opt_policy
    opt_policies = data['arr_2'].astype(np.float32)
    opt_policy = opt_policies[0]
    
    # opt_distances present in the dataset and choosing a single opt_distance
    opt_distances = data['arr_3'].astype(np.float32)
    opt_dist = opt_distances[0]

    start_maps = []
    opt_trajs = []

    opt_dist_flat = opt_dist.flatten()  # flatten the opt_dist array
    opt_dist_vals = opt_dist_flat[opt_dist_flat > opt_dist_flat.min()]  # get the values greater than the minimum value  
    percentage = np.array([0.55, 0.70, 0.85, 1.0])  # percentage of the values to be considered
    opt_distance_threshold = np.percentile(opt_dist_vals, (1-percentage) * 100.0)  # get the threshold values
    random_opt_dist_index = np.random.randint(0, len(opt_distance_threshold)-1)  # get a random index
    start_candidate = (opt_dist_flat > opt_distance_threshold[random_opt_dist_index + 1])&(opt_dist_flat <= opt_distance_threshold[random_opt_dist_index]) # get the start candidate
    start_index = np.random.choice(np.where(start_candidate)[0])  # get a random index from the start candidate
    start_map = np.zeros_like(opt_dist)  # create a start map
    start_map.ravel()[start_index] = 1  # set the start index to 1

    # generating Optimal trajectory from start_map, goal_map and opt_policy
    optimal_traj = np.zeros_like(start_map) # create a zero array of the same shape as start_map
    optimal_policy = opt_policy.transpose((1,2,3,0)) # transpose the opt_policy to match the shape of start_map
    a,b,c = np.nonzero(start_map) # get the indices of the start_map where the value is 1
    current_location = tuple(np.array([a, b, c]).squeeze()) # get the current location based on the indices derived above
    e,f,g = np.nonzero(goal_map) # get the indices of the goal_map where the value is 1
    goal_location = tuple(np.array([e, f, g]).squeeze()) # get the goal location based on the indices derived above
    
    
    # print(current_location, goal_location)
    # print(start_map[a, b, c])
    # print(goal_map[e, f, g])
    # print(np.nonzero(start_map), np.nonzero(goal_map))

    while np.all(goal_location != current_location): # loop until the current location is equal to the goal location
        optimal_traj[current_location] = 1.0 # set the current location to 1.0
        action_to_move = [
            (0,-1,0), # move in the negative x direction
            (0,0,+1), # move in the positive y direction
            (0,0,-1), # move in the negative y direction
            (0,+1,0), # move in the positive x direction
            (0,-1,+1), # move in the negative x direction and positive y direction
            (0,-1,-1), # move in the negative x direction and negative y direction
            (0,+1,+1), # move in the positive x direction and positive y direction
            (0,+1,-1) # move in the positive x direction and negative y direction
        ]
        
        action = optimal_policy[current_location]
        move = action_to_move[np.argmax(action)]
        next_location = np.add(current_location, move)
        x_cord, y_cord, z_cord  = current_location[1], current_location[2], current_location[0]
        new_x_cord = x_cord + move[1]
        new_y_cord = y_cord + move[2]
        new_z_cord = z_cord + move[0]
        # print(x_cord, y_cord, z_cord, new_x_cord, new_y_cord, new_z_cord)

        next_loc = (new_z_cord, new_x_cord, new_y_cord)
        if optimal_traj[next_loc] == 0.0: # check if the node is already visited or not
            current_location = next_loc

# convert the goal map to a tensor
# print(goal_map.shape)
goal_maps = torch.tensor(goal_map)
# print(goal_maps.shape)


def get_heuristic(goal_maps: torch.tensor, tb_factor: float = 0.001) -> torch.tensor:
    """
    Get heuristic function for A* search (chebyshev + small const * euclidean)

    Args:
        goal_maps (torch.tensor): one-hot matrices of goal locations
        tb_factor (float, optional): small constant weight for tie-breaking. Defaults to 0.001.

    Returns:
        torch.tensor: heuristic function matrices
    """

    # some preprocessings to deal with mini-batches
    num_samples, H, W = goal_maps.shape[0], goal_maps.shape[-2], goal_maps.shape[-1]
    print(f"num_samples: {num_samples}, H: {H}, W: {W}")
    print(torch.meshgrid(torch.arange(0, H)))
    exit()
    grid = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    loc = torch.stack(grid, dim=0).type_as(goal_maps)
    loc_expand = loc.reshape(2, -1).unsqueeze(0).expand(num_samples, 2, -1)
    goal_loc = torch.einsum("kij, bij -> bk", loc, goal_maps)
    goal_loc_expand = goal_loc.unsqueeze(-1).expand(num_samples, 2, -1)

    # chebyshev distance
    dxdy = torch.abs(loc_expand - goal_loc_expand)
    h = dxdy.sum(dim=1) - dxdy.min(dim=1)[0]
    euc = torch.sqrt(((loc_expand - goal_loc_expand) ** 2).sum(1))
    h = (h + tb_factor * euc).reshape_as(goal_maps)

    return h

result = get_heuristic(goal_maps) 
print(result.shape)
print(result)

# generate the result as an image


