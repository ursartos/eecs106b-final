from re import T
from casadi import Opti, sin, cos, tan, vertcat, mtimes, floor, conditional, MX, SX, if_else, logic_and

import numpy as np
import matplotlib.pyplot as plt

from dijkstar import Graph, find_path

def position_to_grid(terrains, pos, side_length):
    return (pos/side_length * len(terrains)).astype(int)
    
def xy_to_i(terrain_map, xy):
    return len(terrain_map) * xy[0] + xy[1]

def i_to_xy(terrain_map, i):
    return np.array([i % len(terrain_map), i // len(terrain_map)])

def shortest_path_to_goal(terrains_grid, side_length, start, goal):
    graph = Graph()
    start_node = xy_to_i(terrains_grid, position_to_grid(terrains_grid, start, side_length))
    goal_node = xy_to_i(terrains_grid, position_to_grid(terrains_grid, goal, side_length))

    for i in range(len(terrains_grid)):
        for j in range(len(terrains_grid[0])):
            current_node_idx = xy_to_i(terrains_grid, [i, j])
            
            for r in range(-1, 2):
                for c in range(-1, 2):
                    if r == 0 and c == 0:
                        continue
                    if i + r < 0 or i + r >= len(terrains_grid):
                        continue
                    if j + c < 0 or j + c >= len(terrains_grid[0]):
                        continue
                    neighbor_node_idx = xy_to_i(terrains_grid, [i + r, j + c])
                    min_r, min_c = min(i, i + r), min(j, j + c)
                    graph.add_edge(current_node_idx, neighbor_node_idx, 1/terrains_grid[min_r, min_c, 0])

    pathinfo = find_path(graph, start_node, goal_node)

    real_world_points = []
    for i in range(len(pathinfo)):
        real_world_points.append(i_to_xy(terrains_grid, pathinfo[i]) * side_length / len(terrains_grid))

def plan_to_pose(q_start, q_goal, q_lb, q_ub, u_lb, u_ub, obs_list, horizon=1, dt=0.01, terrain_map=None, side_length=None):
    """
    Plans a path from q_start to q_goal.

    q_lb, q_ub are the state lower and upper bounds repspectively.
    u_lb, u_ub are the input lower and upper bounds repspectively.
    obs_list is the list of obstacles, each given as a tuple (x, y, r).
    L is the length of the car.
    n is the number of timesteps.
    dt is the discretization timestep.

    Returns a plan (shape (3, n+1)) of waypoints and a sequence of inputs
    (shape (2, n)) of inputs at each timestep.
    """
    path = np.array([[0,0], [1,1], [2,2], [3,3], [4,4]])
    waypoints, inputs, n = path_to_trajectory(path, q_start, q_goal, terrain_map, side_length, horizon, dt)
    print(waypoints)
    print(inputs)
    return waypoints, inputs, n

def path_to_trajectory(path, q_start, q_goal, terrain_map, side_length, horizon=1, dt=0.05):
    n = path.shape[0]*horizon
    res = terrain_map.shape[0]/side_length
    c = max(1, path.shape[0]//n) # look ahead this many indexes
    waypoints = np.zeros((n, 3))
    inputs = np.zeros((n-1, 2))
    print(waypoints.shape, inputs.shape)
    waypoints[0] = [path[0,0], path[0,1], q_start[2]]
    i = c
    for j in range(1,waypoints.shape[0]-1):
        # make sure the last cell is the goal position #
        if (i + c > terrain_map.shape[0]):
            c = terrain_map.shape[0] - i
            1/0
        
        # generate waypoint #
        print(i, j, path[i], path[i+c])
        theta = np.arctan(path[i+c,1] - path[i,1])/(path[i+c,0] - path[i,0])
        waypoints[j] = [path[i,0], path[i,1], theta]

        # velocity of last waypoint to get to current waypoint #
        distance = np.sqrt((waypoints[j,1] - waypoints[j-1,1])**2 + (waypoints[j,0] - waypoints[j-1,0])**2)
        theta_d = waypoints[j,2] - waypoints[j-1,2]
        inputs[j-1] = [distance/dt, theta_d/dt]

        i += c
    
    waypoints[-1] = [path[-1,0], path[-1,1], q_goal[2]]
    distance = np.sqrt((waypoints[-1,1] - waypoints[-2,1])**2 + (waypoints[-1,0] - waypoints[-2,0])**2)
    theta_d = waypoints[-1,2] - waypoints[-2,2]
    inputs[-1] = [distance/dt, theta_d/dt]

    assert((waypoints[0] == q_start).all())
    assert((waypoints[-1] == q_goal).all())
    return waypoints, inputs, n



def plot(plan, inputs, times, q_lb, q_ub, obs_list, terrains):

    # Trajectory plot
    ax = plt.subplot(1, 1, 1)
    ax.set_aspect(1)
    ax.set_xlim(q_lb[0], q_ub[0])
    ax.set_ylim(q_lb[1], q_ub[1])

    for obs in obs_list:
        xc, yc, r = obs
        circle = plt.Circle((xc, yc), r, color='black')
        ax.add_artist(circle)
    
    for ter in terrains:
        # print(ter)
        w, h = ter[0][1] - ter[0][0], ter[0][3] - ter[0][2]
        x, y = ter[0][0], ter[0][2]
        # print(x,y,w,h)
        # x, y = 4,4
        rect = plt.Rectangle((x, y), w, h, color='grey')
        ax.add_artist(rect)

    plan_x = plan[0, :]
    plan_y = plan[1, :]
    ax.plot(plan_x, plan_y, color='green')

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.show()

    # States plot
    plt.plot(times, plan[0, :], label='x')
    plt.plot(times, plan[1, :], label='y')
    plt.plot(times, plan[2, :], label='theta')

    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()

    # Inputs plot
    plt.plot(times[:-1], inputs[0, :], label='u1')
    plt.plot(times[:-1], inputs[1, :], label='u2')
    
    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()


def main():
    
    ###### PROBLEM PARAMS ######

    n = 1000
    dt = 0.5

    xy_low = [0, 0]
    xy_high = [4, 4]
    u1_max = 2
    u2_max = 3
    obs_list = []#[[2, 1, 1]]#, [-3, 4, 1], [4, 2, 2]]
    q_start = np.array([0, 0, 0])
    q_goal = np.array([4, 4, 0])

    ###### SETUP PROBLEM ######
    
    q_lb = xy_low
    q_ub = xy_high

    u_lb = [-u1_max, -u2_max]
    u_ub = [u1_max, u2_max]

    ###### CONSTRUCT SOLVER AND SOLVE ######

    terrain1 = ([1, 2, 0, 1], 0.05, 0.05)
    terrains = []#[terrain1]

    res = 1
    horizon = res
    side_length = q_ub[0] - q_lb[0]
    terrain_map = np.ones((res*side_length, res*side_length, 2))
    for terrain in terrains:
        xmin, xmax, ymin, ymax = res*terrain[0]
        k, d = terrain[1:]
        terrain_map[xmin:xmax, ymin:ymax, :] = [k, d]

    plan, inputs, n = plan_to_pose(q_start, q_goal, q_lb, q_ub, u_lb, u_ub, obs_list, horizon=horizon, dt=dt, terrain_map=terrain_map, side_length=side_length)
    plan = plan.T
    inputs = inputs.T

    ###### PLOT ######

    times = np.arange(0.0, n * dt, dt)
    print("Final Position:", plan[:3, -1])
    plot(plan, inputs, times, q_lb, q_ub, obs_list, terrains)

if __name__ == '__main__':
    main()