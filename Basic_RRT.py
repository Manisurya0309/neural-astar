import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import imageio

path =  './planning-datasets/data/mpd/mazes_032_moore_c8.npz'
output_path = './test.npy'

with np.load(path) as data:
    # print("Files in .npz file: , data.files", list(data.keys()))

    # map_designs:   Representations of the environment's layout where navigation occurs. Each design is a 32x32 grid where cells can be either 1 (indicating an obstacle) or 0 (free space).
    # goal_maps:     Indicates goal locations within the environment. Each map is a 32x32 grid where a cell with a value of 1 marks the goal location, and 0s are free spaces.
    # opt_policies:  Stores optimal policies/actions for each position in the environment, given as probabilities or choices across 8 possible actions from each grid cell.
    # opt_distances: Contains the distance from each cell in the environment to the goal location.

    map_designs = data['arr_0'].astype(np.float32)   # 800, 32, 32        # numpy.ndarray
    goal_maps = data['arr_1'].astype(np.float32)     # 800, 1, 32, 32     # numpy.ndarray
    opt_policies = data['arr_2'].astype(np.float32)  # 800, 8, 1 , 32, 32 # numpy.ndarray
    opt_distances = data['arr_3'].astype(np.float32) # 800, 1, 32, 32     # numpy.ndarray
    
    num_actions = opt_policies.shape[1] # 8
    num_orientations = opt_policies.shape[2] # 1
    
    map_design = map_designs[0][np.newaxis]          # it is a 3D array of shape (1, 32, 32) with values 1 and 0
    goal_map = goal_maps[0]                          # it is a 3D array of shape (1, 32, 32) with values 1 and 0 where 1 is 1 is the goal location and 0 is the free space
    opt_policy = opt_policies[0]                     # it is a 3D array of shape (8, 1, 32, 32) with values between 0 and 1
    opt_dist = opt_distances[0]                      # it is a 3D array of shape (1, 32, 32) with values of distance from the goal location to that node
    start_maps, opt_trajs = [], []                   # list to store the start maps and optimal trajectories      
    
    # generating start map from 'opt_dist'
    od_vct = opt_dist.flatten() # 1024
    od_vals = od_vct[od_vct > od_vct.min()] # 517 
    pct = np.array([0.55,0.70,0.85,1.0])
    od_th = np.percentile(od_vals, 100.0*(1-pct))
    r = np.random.randint(0, len(od_th)-1)
    start_candidate = (od_vct >= od_th[r+1])&(od_vct <= od_th[r])

    start_idx = np.random.choice(np.where(start_candidate)[0])
    start_map = np.zeros_like(opt_dist)
    start_map.ravel()[start_idx] = 1

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # generating optimal trajectory from 'start_map', 'goal_map', 'opt_policy'
    opt_traj = np.zeros_like(start_map)
    opt_policy = opt_policy.transpose((1,2,3,0))
    current_loc = tuple((np.array(np.nonzero(start_map)).squeeze()))
    goal_loc = tuple((np.array(np.nonzero(goal_map)).squeeze()))
    while goal_loc != current_loc:
        opt_traj[current_loc] = 1.0
        action_to_move = [
        (0, -1, 0),
        (0, 0, +1),
        (0, 0, -1),
        (0, +1, 0),
        (0, -1, +1),
        (0, -1, -1),
        (0, +1, +1),
        (0, +1, -1),
    ]
        move = action_to_move[np.argmax(opt_policy[current_loc])]
        next_loc = np.add(current_loc, move)
        next_loc = tuple(next_loc)
        assert (
            opt_traj[next_loc] == 0.0
        )
        current_loc = next_loc

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    start_maps.append(start_map)
    start_map = np.concatenate(start_maps)
    opt_trajs.append(opt_traj)
    opt_traj = np.concatenate(opt_trajs)

    # convert the above map into a cost map using mahattan distance from the goal location and start location
    start_map = start_map.squeeze()
    goal_map = goal_map.squeeze()
    opt_traj = opt_traj.squeeze()
    opt_dist = opt_dist.squeeze()
    

    # build an path using the start_map, goal_map and map_designs using RRT
    class RRT:
        def __init__(self, start_map, goal_map, map_designs):
            self.start_map = start_map.squeeze()
            self.goal_map = goal_map.squeeze()
            self.map_design = map_designs[0].squeeze()
            self.frames = []
            
        
        def build_path(self):
            start_map = self.start_map
            goal_map = self.goal_map
            map_design = self.map_design
            
            # convert the map to coordinates
            start_pos = np.argwhere(start_map == 1)[0]
            goal_pos = np.argwhere(goal_map == 1)[0]
            # print(f"start_pos : {start_pos}")
            # print(f"goal_pos : {goal_pos}")

            #  initialise the RRT tree as list of Nodes where each node is a dict with current position and its parent index
            self.tree = [{"pos": start_pos, "parent": None}]

            self.capture_frame() # capture the current state of the RRT and save as an image frame

            # loop until the goal is reached
            for i in range(1000):
                rand_position = self.sample_random_position()
                nearest_node_index = self.find_nearest_node_index(rand_position)
                new_pos = self.steer(rand_position, nearest_node_index)
                print(f"rand_position : {rand_position}",f"nearest_node_index : {nearest_node_index}",f"new_pos : {new_pos}" )
                if self.check_collision(new_pos, nearest_node_index):
                    continue
                self.tree.append({"pos": new_pos, "parent": nearest_node_index})
                self.capture_frame() # capture the current state of the RRT and save as an image frame
                if np.linalg.norm(new_pos - goal_pos) < 2:
                    print("goal reached")
                    break
            self.capture_frame() # capture the final state of the RRT and save as an image frame

        def sample_random_position(self):
            # implementing the random sampling of the position in map_design
            while True:
                rand_pos = np.random.randint(0, self.map_design.shape[0], 2)
                if self.map_design[rand_pos[0], rand_pos[1]] == 1:
                    return rand_pos
                
        def find_nearest_node_index(self, rand_pos):
            # find the nearest node in the tree to the rand_pos
            nearest_node_index = None
            min_dist = float('inf')
            for index, node in enumerate(self.tree):
                dist = np.linalg.norm(np.array(node["pos"]) - np.array(rand_pos))
                if dist < min_dist:
                    min_dist = dist
                    nearest_node_index = index
            return nearest_node_index
        
        def steer(self, rand_pos, nearest_node_index):
            # steer the nearest node towards the rand_pos
            nearest_node_pos = self.tree[nearest_node_index]["pos"]
            direction = rand_pos - nearest_node_pos
            dist = np.linalg.norm(direction)
            step_size = min(1, dist)
            if dist == 0:
                step = direction
            else:
                step = (direction / dist) * step_size
            new_pos = self.tree[nearest_node_index]["pos"] + step
            return np.round(new_pos).astype(int)
        
        def check_collision(self, new_pos, nearest_node_index):
            nearest_node_pos = self.tree[nearest_node_index]['pos']
            line_dist = np.linalg.norm(new_pos - nearest_node_pos)
            n_steps = int(np.ceil(line_dist))
            
            x, y = np.linspace(nearest_node_pos[0], new_pos[0], n_steps), np.linspace(nearest_node_pos[1], new_pos[1], n_steps)
            

            for x_step, y_step in zip(x, y):
                x_index, y_index = int(round(x_step)), int(round(y_step))

                if self.map_design[x_index, y_index] == 0:
                    return True  
            
            return False  
        
        def calculate_node_heuristic(node_position, goal_position, tb_factor=0.001):
            """
            Calculate heuristic for a single node in RRT based on Chebyshev and Euclidean distances to the goal.

            Args:
                node_position (np.array): The (x, y) coordinates of the current node.
                goal_position (np.array): The (x, y) coordinates of the goal node.
                tb_factor (float, optional): Small constant weight for tie-breaking. Defaults to 0.001.

            Returns:
                float: Heuristic value for the node.
            """
            node_position_tensor = torch.tensor(node_position, dtype=torch.float32)
            goal_position_tensor = torch.tensor(goal_position, dtype=torch.float32)
            
            dxdy = torch.abs(node_position_tensor - goal_position_tensor)
            chebyshev = torch.max(dxdy)
            euclidean = torch.norm(node_position_tensor - goal_position_tensor)
            
            heuristic_value = chebyshev + tb_factor * euclidean
            
            return heuristic_value.item()

        
        def capture_frame(self):
            # Capture the current state of the RRT and save as an image frame
            fig, ax = plt.subplots()
            ax.imshow(self.map_design, cmap='Greys', origin='lower')
            start_pos = np.argwhere(self.start_map == 1)[0]
            goal_pos = np.argwhere(self.goal_map == 1)[0]
            ax.plot(start_pos[1], start_pos[0], 'go', markersize=10)  # Start
            ax.plot(goal_pos[1], goal_pos[0], 'ro', markersize=10)  # Goal
            for node in self.tree:
                if node['parent'] is not None:
                    parent_pos = self.tree[node['parent']]['pos']
                    ax.plot([node['pos'][1], parent_pos[1]], [node['pos'][0], parent_pos[0]], 'b-')
            # Hide axis labels
            ax.set_xticks([])
            ax.set_yticks([])
            # Save the current frame
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.frames.append(image)
            plt.close(fig)

        def save_gif(self, filename='rrt_path.gif', fps=10):
            
            if not os.path.exists('RRT_output'):
                os.makedirs('RRT_output')

            filename = os.path.join('RRT_output', filename)
            imageio.mimsave('RRT_output', self.frames, fps=fps)


    rrt = RRT(start_map, goal_map, map_designs)
    rrt.build_path()
    rrt.save_gif('rrt_path_planning.gif')



