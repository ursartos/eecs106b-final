#!/usr/bin/env python
"""
Starter code for EECS C106B Spring 2022 Project 2.
Author: Valmik Prabhu, Amay Saxena

Implements the optimization-based path planner. You shouldn't need to edt this file.
Just place your completed optimization_planner_casadi.py file in the planners/ folder 
(same folder as this file). It should then work out of the box.
"""

import scipy as sp
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt

from .configuration_space import UnicycleConfigurationSpace, Plan, expanded_obstacles
# from .optimization_planner_casadi import plan_to_pose
from .optimization_planner_dijkstra import plan_to_pose

def make_plan(q_opt, u_opt, N, dt):
    times = []
    target_positions = []
    open_loop_inputs = []
    t = 0

    # print(q_opt.shape)
    # print(u_opt.shape)
    # print(N)
    q_opt = q_opt.T
    u_opt = u_opt.T

    for i in range(0, N-1):
        qi = np.array([q_opt[0][i], q_opt[1][i], q_opt[2][i]])
        ui = np.array([u_opt[0][i], u_opt[1][i]])
        times.append(t)
        target_positions.append(qi)
        open_loop_inputs.append(ui)
        t = t + dt

    # We add one extra step since q_opt has one more state that u_opt
    qi = np.array([q_opt[0][N-1], q_opt[1][N-1], q_opt[2][N-1]])
    ui = np.array([0.0, 0.0])
    times.append(t)
    target_positions.append(qi)
    open_loop_inputs.append(ui)

    return Plan(np.array(times), np.array(target_positions), np.array(open_loop_inputs), dt)

class OptimizationPlanner(object):
    def __init__(self, config_space, gt_terrains, mas, all_plans):
        self.config_space = config_space

        self.input_low_lims = self.config_space.input_low_lims
        self.input_high_lims = self.config_space.input_high_lims
        self.gt_terrains = gt_terrains
        self.optimal_plan = None
        self.true_cost = None
        self.optimal_cost = None
        self.all_plans = all_plans

    def plan_to_pose(self, start, goal, dt=0.01, N=1000):
        """
            Uses your optimization based path planning algorithm to plan from the 
            start configuration to the goal configuration.
            This function interfaces with your python code from homework.
            
            Please place your completed optimization_planner_casadi.py file in the 
            planners/ folder (same folder as this file).

            Args:
                start: starting configuration of the robot.
                goal: goal configuration of the robot.
                dt: Discretization time step. How much time we would like between
                    subsequent time-stamps.
                N: How many waypoints would we like to have in our path from start
                   to goal
        """

        print("======= Planning with OptimizationPlanner =======")

        # Expand obstacles to account for the radius of the robot.
        with expanded_obstacles(self.config_space.obstacles, self.config_space.robot_radius + 0.05):

            self.plan = None

            # q_opt, u_opt = plan_to_pose(np.array(start), np.array(goal), 
            #     self.config_space.low_lims, self.config_space.high_lims, 
            #     self.input_low_lims, self.input_high_lims, self.config_space.obstacles, 
            #     n=N, dt=dt, terrains=self.config_space.terrains)

            q_opt, u_opt, N, proposed_plan, _, _ = plan_to_pose(np.array(start), np.array(goal), 
                self.config_space.low_lims, self.config_space.high_lims, 
                self.input_low_lims, self.input_high_lims, self.config_space.obstacles, 
                dt=dt, terrain_map=self.config_space.terrains)

            q_opt_gt, u_opt_gt, N_gt, _, self.true_cost, self.optimal_cost = plan_to_pose(np.array(start), np.array(goal), 
                self.config_space.low_lims, self.config_space.high_lims, 
                self.input_low_lims, self.input_high_lims, self.config_space.obstacles, 
                dt=dt, terrain_map=self.gt_terrains, proposed_plan=proposed_plan)

            self.plan = make_plan(q_opt, u_opt, N, dt)
            self.optimal_plan = make_plan(q_opt_gt, u_opt_gt, N_gt, dt)

        return self.plan

    def plot_execution(self):
        """
        Creates a plot of the planned path in the environment. Assumes that the 
        environment of the robot is in the x-y plane, and that the first two
        components in the state space are x and y position. Also assumes 
        plan_to_pose has been called on this instance already.
        """

        ax = plt.subplot(1, 1, 1)
        ax.set_aspect(1)
        ax.set_xlim(self.config_space.low_lims[0], self.config_space.high_lims[0])
        ax.set_ylim(self.config_space.low_lims[1], self.config_space.high_lims[1])

        for obs in self.config_space.obstacles:
            xc, yc, r = obs
            circle = plt.Circle((xc, yc), r, color='black')
            ax.add_artist(circle)

        for terrain in self.config_space.terrains:
            x1, x2 = terrain[0]
            y1, y2 = terrain[1]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1)
            ax.add_artist(rect)

        if self.plan:
            self.all_plans.append(self.plan)
            for plan in self.all_plans:
                plan_x = plan.positions[:, 0]
                plan_y = plan.positions[:, 1]
                ax.plot(plan_x, plan_y)

        plt.show()
        return self.plan

def main():
    """Use this function if you'd like to test without ROS.
    """
    # start = np.array([1, 1, 0]) 
    # goal = np.array([9, 9, 0])
    # xy_low = [0, 0]
    # xy_high = [10, 10]
    # phi_max = 0.6
    # u1_max = 2
    # u2_max = 3
    # obstacles = []#[[6, 3.5, 1.5], [3.5, 6.5, 1]]

    # terrain1 = ([1, 9, 4, 10], 0.05, 0.05)
    # terrains = [terrain1]

    # res = 3
    # horizon = max(1, res//2)
    # side_length = xy_high[0] - xy_low[0] + 1
    # terrain_map = np.ones((res*side_length, res*side_length, 2))
    # for terrain in terrains:
    #     xmin, xmax, ymin, ymax = [res*x for x in terrain[0]]
    #     k, d = terrain[1:]
    #     terrain_map[xmin:xmax, ymin:ymax, :] = [k, d]

    ###### PROBLEM PARAMS ######

    n = 1000
    dt = 0.5

    xy_low = [0, 0]
    xy_high = [40, 40]
    u1_max = 2
    u2_max = 3
    obs_list = []#[[2, 1, 1]]#, [-3, 4, 1], [4, 2, 2]]
    q_start = np.array([0, 0, 0])
    q_goal = np.array([23, 21, 0])

    ###### SETUP PROBLEM ######
    
    q_lb = xy_low
    q_ub = xy_high

    u_lb = [-u1_max, -u2_max]
    u_ub = [u1_max, u2_max]

    ###### CONSTRUCT SOLVER AND SOLVE ######

    terrain1 = ([1, 9, 4, 10], 0.05, 0.05)
    terrains = [terrain1]

    res = 3
    horizon = max(1, res//2)

    side_length = q_ub[0] - q_lb[0] + 1
    terrain_map = np.ones((res*side_length, res*side_length, 2))
    for terrain in terrains:
        xmin, xmax, ymin, ymax = [res*x for x in terrain[0]]
        k, d = terrain[1:]
        terrain_map[xmin:xmax, ymin:ymax, :] = [k, d]

    config = UnicycleConfigurationSpace(xy_low + [-1000],
                                        xy_high + [1000],
                                        [-u1_max, -u2_max],
                                        [u1_max, u2_max],
                                        obs_list,
                                        0.15,
                                        q_start,
                                        q_goal,
                                        terrains=terrain_map
                                        )

    planner = OptimizationPlanner(config)
    plan = planner.plan_to_pose(q_start, q_goal)
    planner.plot_execution()

if __name__ == '__main__':
    main()
