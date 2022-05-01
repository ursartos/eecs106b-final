#!/usr/bin/env python
"""
Starter code for EE106B Turtlebot Lab
Author: Valmik Prabhu, Chris Correa
Adapted for Spring 2020 by Amay Saxena
"""
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt

from std_srvs.srv import Empty as EmptySrv
import rospy

from proj2_pkg.msg import UnicycleCommandMsg, UnicycleStateMsg
from proj2.planners import RRTPlanner, OptimizationPlanner, UnicycleConfigurationSpace #Sinusoid_Planner
from proj2.controller import UnicycleModelController

def parse_args():
    """
    Pretty self explanatory tbh
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-planner', '-p', type=str, default='opt', help=
        'Options: sin, rrt, opt.  Default: opt')
    parser.add_argument('-x', type=float, default=1.0, help='Desired position in x')
    parser.add_argument('-y', type=float, default=1.0, help='Desired position in y')
    parser.add_argument('-theta', type=float, default=0.0, help='Desired turtlebot angle')
    return parser.parse_args()

def get_terrain_map(terrains, xy_low, xy_high, res=1):
    # this function is mocking the distribution of terrains features on the ground spatially
    terrains = np.array(terrains)
    side_length = xy_high[0] - xy_low[0] + 1
    terrain_map = np.zeros((res*side_length, res*side_length, len(terrains) + 1))
    terrain_map[:, :, 0] = 1
    for i, terrain in enumerate(terrains):
        # print(terrain[0])
        xmin, xmax, ymin, ymax = [res*x for x in terrain[0]]
        encoding = np.zeros((len(terrains) + 1,))
        encoding[i+1] = 1
        terrain_map[xmin:xmax, ymin:ymax, :] = encoding
    return terrain_map

def get_terrain_kd(terrain_map, controller):
    kd_map = np.ones(terrain_map.shape[:2] + (2,))
    kd_aleatoric_map = np.zeros(terrain_map.shape[:2] + (2,))
    kd_epistemic_map = np.zeros(terrain_map.shape[:2] + (2,))
    for i in range(terrain_map.shape[0]):
        for j in range(terrain_map.shape[1]):
            terrain_input = np.array([terrain_map[i, j]])
            d, d_uncertainty, d_aleatoric = controller.d_estimator.predict(terrain_input)
            k, k_uncertainty, k_aleatoric = controller.k_estimator.predict(terrain_input)
            kd_map[i, j] = [max(0.01, d), max(0.01, k)]
            kd_aleatoric_map[i, j] = [max(0., d_aleatoric), max(0., k_aleatoric)]
            kd_epistemic_map[i, j] = [max(0., d_uncertainty), max(0, k_uncertainty)]
    return kd_map, kd_aleatoric_map, kd_epistemic_map, np.max((np.zeros(kd_map.shape), kd_map - np.sqrt(kd_aleatoric_map)), axis=0)

# def get_terrain_image(filename='/path/to/file'):

def compute_estimator_error(estimate, truth_lst):
    truth = np.ones(estimate.shape)
    print(truth_lst)
    print(estimate)
    for terr in truth_lst:
        x_low, x_high, y_low, y_high = terr[0]
        for x in range(x_low, x_high):
            for y in range(y_low, y_high):
                truth[x,y] = terr[1][0]

    average_error = 0
    errors = []
    for i in range(estimate.shape[0]):
        for j in range(estimate.shape[1]):
            if (estimate[i,j] == 1.0):
                continue

            error = abs(estimate[i,j] - truth[i,j])
            errors.append(error)
            if (average_error == 0):
                average_error = error
            else:
                average_error = (average_error + error)/2

    return average_error, errors
    

if __name__ == '__main__':
    rospy.init_node('planning', anonymous=False)
    args = parse_args()

    # reset state
    print('Waiting for converter/reset service ...')
    rospy.wait_for_service('/converter/reset')
    print('found!')
    reset = rospy.ServiceProxy('/converter/reset', EmptySrv)
    reset()

    if not rospy.has_param("/environment/obstacles"):
        raise ValueError("No environment information loaded on parameter server. Did you run init_env.launch?")
    obstacles = rospy.get_param("/environment/obstacles")

    if not rospy.has_param("/environment/terrains"):
        raise ValueError("No environment information loaded on parameter server. Did you run init_env.launch?")
    terrains = rospy.get_param("/environment/terrains")

    if not rospy.has_param("/environment/low_lims"):
        raise ValueError("No environment information loaded on parameter server. Did you run init_env.launch?")
    xy_low = rospy.get_param("/environment/low_lims")

    if not rospy.has_param("/environment/high_lims"):
        raise ValueError("No environment information loaded on parameter server. Did you run init_env.launch?")
    xy_high = rospy.get_param("/environment/high_lims")

    # if not rospy.has_param("/bicycle_converter/converter/max_steering_angle"):
    #     raise ValueError("No robot information loaded on parameter server. Did you run init_env.launch?")
    # phi_max = rospy.get_param("/bicycle_converter/converter/max_steering_angle")

    if not rospy.has_param("/unicycle_converter/converter/max_steering_rate"):
        raise ValueError("No robot information loaded on parameter server. Did you run init_env.launch?")
    u2_max = rospy.get_param("/unicycle_converter/converter/max_steering_rate")

    if not rospy.has_param("/unicycle_converter/converter/max_linear_velocity"):
        raise ValueError("No robot information loaded on parameter server. Did you run init_env.launch?")
    u1_max = rospy.get_param("/unicycle_converter/converter/max_linear_velocity")

    u1_max = 0.5

    terrain_map_res = 1
    controller = UnicycleModelController()
    rospy.sleep(1)

    start = np.array(controller.state) 
    goal = np.array([args.x, args.y, args.theta])

    terrain_visual_features = get_terrain_map(terrains, xy_low, xy_high, terrain_map_res)
    raw_terrain_map, terrain_aleatoric_map, terrain_epistemic_map, terrain_map = get_terrain_kd(terrain_visual_features, controller)

    args.planner = 'opt'

    if args.planner == 'sin':
        raise ValueError("don't use sin, just don't")
        planner = SinusoidPlanner(config)
        ## Edit the dt and delta_t arguments to your needs.
        plan = planner.plan_to_pose(controller.state, goal, dt=0.01, delta_t=2.0)

    elif args.planner == 'rrt':
        ## Edit the max_iter, expand_dist, dt and prefix_time_length arguments to your needs.
        planner = RRTPlanner(config, max_iter=5000, expand_dist=1.6)
        plan = planner.plan_to_pose(controller.state, goal, dt=0.05, prefix_time_length=0.05)

    elif args.planner == 'opt':
        goals = [start, goal]
        counter = 1
        while True:
            plt.imshow(raw_terrain_map[:, :, 0])
            plt.title("Raw Terrain D Map")
            plt.colorbar()
            plt.show()
            plt.imshow(terrain_aleatoric_map[:, :, 0])
            plt.title("Terrain D Aleatoric Uncertainty Map")
            plt.colorbar()
            plt.show()
            plt.imshow(terrain_epistemic_map[:, :, 0])
            plt.title("Terrain D Epistemic Uncertanity Map")
            plt.colorbar()
            plt.show()
            ## Edit the dt and N arguments to your needs.
            config = UnicycleConfigurationSpace(xy_low,
                                        xy_high,
                                        [-u1_max, -u2_max],
                                        [u1_max, u2_max],
                                        obstacles,
                                        0.15,start,goal,
                                        terrains=terrain_map)
            planner = OptimizationPlanner(config)

            start = controller.state
            plan = planner.plan_to_pose(start, goal, dt=0.01, N=800)
            
            print("Predicted Initial State")
            print(plan.start_position())
            print("Predicted Final State")
            print(plan.end_position())

            planner.plot_execution()

            controller.execute_plan(plan, terrain_vectors=terrain_visual_features,
                                          terrain_map=terrain_map, terrain_map_res=terrain_map_res,
                                          mock_terrains=terrains)
            print("Final State")
            print(controller.state)

            counter += 1
            goal = goals[counter % 2]

            raw_terrain_map, terrain_aleatoric_map, terrain_epistemic_map, terrain_map = get_terrain_kd(terrain_visual_features, controller)
            config.terrains = terrain_map
            print("estimator error", compute_estimator_error(terrain_map[:,:,0], terrains))
