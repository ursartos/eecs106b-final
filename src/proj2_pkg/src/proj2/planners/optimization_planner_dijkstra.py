from re import T
# from casadi import Opti, sin, cos, tan, vertcat, mtimes, floor, conditional, MX, SX, if_else, logic_and

import numpy as np
import matplotlib.pyplot as plt

from dijkstar import Graph, find_path
from collections import defaultdict
from heapq import *

def position_to_grid(terrains, pos, side_length):
    print("pose", pos, side_length, terrains.shape, pos/float(side_length) * terrains.shape[0])
    return (np.round(pos/float(side_length) * terrains.shape[0])).astype(int)
    
def xy_to_i(terrain_map, xy):
    return terrain_map.shape[0] * xy[0] + xy[1]

def i_to_xy(terrain_map, i):
    return np.array([i // terrain_map.shape[0], i % terrain_map.shape[0]])

# dijkstra custom: https://gist.github.com/kachayev/5990802
def dijkstra(edges, f, t):
    g = defaultdict(list)
    for l,r,c in edges:
        g[l].append((c,r))

    q, seen, mins = [(0,f,())], set(), {f: 0}
    while q:
        (cost,v1,path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t: return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 in seen: continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf"), None

def get_d_for_coords(terrains_grid, point1, point2, debug=True):
    r = point2[0] - point1[0]
    c = point2[1] - point1[1]
    worst_case_d = float('inf')
    bottom_left = np.min((point1, point2), axis=0)
    bottom_left = tuple(bottom_left)
    if abs(r) == 1 and c == 0:
        if bottom_left[1] > 0:
            worst_case_d = min(worst_case_d, terrains_grid[bottom_left[0]][bottom_left[1] - 1][0])
        worst_case_d = min(worst_case_d, terrains_grid[bottom_left][0])
    elif abs(c) == 1 and r == 0:
        if bottom_left[0] > 0:
            worst_case_d = min(worst_case_d, terrains_grid[bottom_left[0] - 1][bottom_left[1]][0])
        worst_case_d = min(worst_case_d, terrains_grid[bottom_left][0])
    elif abs(r) == 1 and abs(c) == 1:
        worst_case_d = min(worst_case_d, terrains_grid[bottom_left][0])
    if debug:
        print("Worst case d", worst_case_d, "between", point1, "and", point2, "bottom left", bottom_left)
    return worst_case_d

def shortest_path_to_goal(terrains_grid, side_length, start, goal, proposed_plan):
    graph = Graph()
    edges = []
    start_node = xy_to_i(terrains_grid, position_to_grid(terrains_grid, start, side_length))
    goal_node = xy_to_i(terrains_grid, position_to_grid(terrains_grid, goal, side_length))
    print("Start", position_to_grid(terrains_grid, start, side_length), "Goal", position_to_grid(terrains_grid, goal, side_length))

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
                    worst_case_d = get_d_for_coords(terrains_grid, [i, j], [i + r, j + c], debug=False)

                    graph.add_edge(current_node_idx, neighbor_node_idx, float(np.sqrt(abs(r) + abs(c))) / worst_case_d)
                    edges.append((current_node_idx, neighbor_node_idx, float(np.sqrt(abs(r) + abs(c))) / worst_case_d))

    use_custom_dijkstra = False

    if use_custom_dijkstra:
        cost, path = dijkstra(edges, start_node, goal_node)
        nodes = []
        while len(path) > 0:
            nodes.append(path[0])
            path = path[1]
        path = nodes
    else:
        pathinfo = find_path(graph, start_node, goal_node)
        print("PATHINFO", pathinfo[0])
        path = pathinfo[0]
        true_cost = 0
        optimal_cost = 0
        if (proposed_plan is not None):
            # print(graph)
            for i in range(len(proposed_plan) -1):
                cur_node = proposed_plan[i]
                next_node = proposed_plan[i+1]
                print(cur_node, next_node)
                true_cost += graph[cur_node][next_node]
            print("true cost", true_cost)
            optimal_cost = pathinfo[3]

    real_world_points = []
    indices = []
    # print(path)
    for node in path:
        real_world_points.append(i_to_xy(terrains_grid, node) * side_length / float(len(terrains_grid)))
        # print("Added real world point", i_to_xy(terrains_grid, node), i_to_xy(terrains_grid, node) * side_length / float(len(terrains_grid)))
        indices.append(i_to_xy(terrains_grid, node))
    print("Created trajectory")

    return path, np.asarray(real_world_points), np.array(indices), true_cost, optimal_cost

def plan_to_pose(q_start, q_goal, q_lb, q_ub, u_lb, u_ub, obs_list, N=1000, dt=0.01, density=100, terrain_map=None, side_length=None, proposed_plan=None):
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
    density = 150
    # path = np.array([[0,0], [1,1], [2,2], [3,3], [4,4]])
    if (side_length is None):
        side_length = q_ub[0] - q_lb[0]
    node_path, path, indices, true_cost, optimal_cost = shortest_path_to_goal(terrain_map, side_length, q_start, q_goal, proposed_plan)
    print("path", path)
    waypoints, inputs, n = path_to_trajectory(path, indices, q_start, q_goal, terrain_map, side_length, density, dt)
    return waypoints, inputs, n, node_path, true_cost, optimal_cost

def path_to_trajectory(path, indices, q_start, q_goal, terrain_map, side_length, density=100, dt=0.05):
    # n = int((path.shape[0]-1)*density+1)
    # waypoints = np.zeros((n, 3))
    # inputs = np.zeros((n-1, 2))

    waypoints = []
    inputs = []

    # start pose #
    waypoints.append([path[0,0], path[0,1], q_start[2]])

    for j in range(0, path.shape[0] - 1):
        current_d = get_d_for_coords(terrain_map, indices[j], indices[j+1])
        # make sure the last cell is the goal position #
        theta = np.arctan2(path[j+1,1] - path[j,1], path[j+1,0] - path[j,0])
        dist = np.linalg.norm(path[j+1] - path[j])
        segment_density = dist*(density + 1)/float(current_d)
        for step in range(1, int(segment_density)):
            waypoints.append(np.concatenate((path[j] * (1 - step/float(segment_density)) + path[j+1] * (step/float(segment_density)), [theta])))
            inputs.append([0, 0])
    # assert((waypoints[0] == q_start).all())
    # assert((waypoints[-1] == q_goal).all())
    return np.array(waypoints), np.array(inputs), len(waypoints)


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
    xy_high = [6, 6]
    u1_max = 2
    u2_max = 3
    obs_list = []#[[2, 1, 1]]#, [-3, 4, 1], [4, 2, 2]]
    q_start = np.array([0, 0, 0])
    q_goal = np.array([5, 5, 0])

    ###### SETUP PROBLEM ######
    
    q_lb = xy_low
    q_ub = xy_high

    u_lb = [-u1_max, -u2_max]
    u_ub = [u1_max, u2_max]

    ###### CONSTRUCT SOLVER AND SOLVE ######

    # terrain1 = ([1, 9, 4, 10], 0.05, 0.05)
    # terrains = [terrain1]
    terrains = [[[2, 3, 2, 3], 0.05, 0.05]]

    res = 5
    side_length = q_ub[0] - q_lb[0] + 1
    terrain_map = np.ones((res*side_length, res*side_length, 2))
    for terrain in terrains:
        xmin, xmax, ymin, ymax = [res*x for x in terrain[0]]
        d, k = terrain[1:]
        terrain_map[xmin:xmax, ymin:ymax, :] = [d, k]

    plan, inputs, n = plan_to_pose(q_start, q_goal, q_lb, q_ub, u_lb, u_ub, obs_list, N=1000, dt=dt, terrain_map=terrain_map, side_length=side_length)
    plan = plan.T
    inputs = inputs.T

    ###### PLOT ######

    times = np.arange(0.0, n * dt, dt)
    print("Final Position:", plan[:3, -1])
    plot(plan, inputs, times, q_lb, q_ub, obs_list, terrains)

if __name__ == '__main__':
    main()