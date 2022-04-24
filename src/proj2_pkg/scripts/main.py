#!/usr/bin/env python
"""
Starter code for EE106B Turtlebot Lab
Author: Valmik Prabhu, Chris Correa
Adapted for Spring 2020 by Amay Saxena
"""
import numpy as np
import sys
import argparse

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

    print("INPUT LIMITS FROM ROS", u1_max, u2_max)
    u1_max = 0.5

    print("Obstacles:", obstacles)
    
    controller = UnicycleModelController()

    rospy.sleep(1)

    print("Initial State")
    print(controller.state)

    start = np.array([1, 1, 0]) 
    goal = np.array([args.x, args.y, args.theta])

    config = UnicycleConfigurationSpace( xy_low,
                                        xy_high,
                                        [-u1_max, -u2_max],
                                        [u1_max, u2_max],
                                        obstacles,
                                        0.15,start,goal,terrains=terrains)
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
        planner = OptimizationPlanner(config)
        ## Edit the dt and N arguments to your needs.
        plan = planner.plan_to_pose(controller.state, goal, dt=0.5, N=300)
    
    print("Predicted Initial State")
    print(plan.start_position())
    print("Predicted Final State")
    print(plan.end_position())

    planner.plot_execution()

    controller.execute_plan(plan)
    print("Final State")
    print(controller.state)
