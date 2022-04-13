#!/usr/bin/env python

"""
Starter code for EECS C106B Spring 2020 Project 2.
Author: Amay Saxena
"""
import sys
import time
#import rospy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from configuration_space import FreeEuclideanSpace, UnicycleConfigurationSpace, Plan

class RRTGraph(object):

    def __init__(self, *nodes):
        self.nodes = [n for n in nodes]
        self.parent = defaultdict(lambda: None)
        self.path = defaultdict(lambda: None)

    def add_node(self, new_config, parent, path):
        new_config = tuple(new_config)
        parent = tuple(parent)
        self.nodes.append(new_config)
        self.parent[new_config] = parent
        self.path[(parent, new_config)] = path

    def get_edge_paths(self):
        for pair in self.path:
            yield self.path[pair]

    def construct_path_to(self, c):
        c = tuple(c)
        print("")
        #print(f"c = {c}")
        #print(f"self.parent[c] = {self.parent[c]}")
        
        return Plan.chain_paths(self.construct_path_to(self.parent[c]), self.path[(self.parent[c], c)]) if self.parent[c] else None

class RRTPlanner(object):

    def __init__(self, config_space, max_iter=10000, expand_dist=0.3):
        # config_space should be an object of type ConfigurationSpace
        # (or a subclass of ConfigurationSpace).
        self.config_space = config_space
        # Maximum number of iterations to run RRT for:
        self.max_iter = max_iter
        # Exit the algorithm once a node is sampled within this 
        # distance of the goal:
        self.expand_dist = expand_dist


    def plan_to_pose(self, start, goal, dt=0.01, prefix_time_length=0.05):
        """
            Uses the RRT algorithm to plan from the start configuration
            to the goal configuration.
        """
        print("======= Planning with RRT =======")
        self.graph = RRTGraph(start)
        self.plan = None
        print("Iteration:", 0)
        for it in range(self.max_iter):
            #sys.stdout.write("\033[F")
            print(" ")
            print("Iteration:", it + 1)
            # if rospy.is_shutdown():
            #     print("Stopping path planner.")
            #     break
            rand_config = self.config_space.sample_config(goal,counter = it)
            #print(f"found random config: {rand_config}")

            if self.config_space.check_collision(rand_config):
                continue
            #print(f"random config didn't have a collision")

            closest_config = self.config_space.nearest_config_to(self.graph.nodes, rand_config)
            #print(f"there are {len(self.graph.nodes)} nodes availible")
            #print(f"nodes on the graph: {self.graph.nodes}")
            #print(f"found closest_config: {closest_config}")

            path = self.config_space.local_plan(closest_config, rand_config)
            #print(f"the chosen path was: ")
            #path.print()
            if self.config_space.check_path_collision(path):
                #print(f"and it had a collision: ")
                continue

            
            delta_path = path.get_prefix(prefix_time_length)
            #print(f"the chosen prefix was: ")
            #delta_path.print()
            new_config = delta_path.end_position()
            self.graph.add_node(new_config, closest_config, delta_path)
            print("we added a new node to the graph")
            if self.config_space.distance(new_config, goal) <= self.expand_dist:
                path_to_goal = self.config_space.local_plan(new_config, goal)
                if self.config_space.check_path_collision(path_to_goal):
                    continue
                print("we have found the goal and there are no collisions")
                self.graph.add_node(goal, new_config, path_to_goal)
                #print(f"we have {len(self.graph.nodes)} nodes!")
                self.plan = self.graph.construct_path_to(goal)
                return self.plan
        print("Failed to find plan in allotted number of iterations.")
        return None

    def plot_execution(self):
        """
        Creates a plot of the RRT graph on the environment. Assumes that the 
        environment of the robot is in the x-y plane, and that the first two
        components in the state space are x and y position. Also assumes 
        plan_to_pose has been called on this instance already, so that self.graph
        is populated. If planning was successful, then self.plan will be populated 
        and it will be plotted as well.
        """
        ax = plt.subplot(1, 1, 1)
        ax.set_aspect(1)
        ax.set_xlim(self.config_space.low_lims[0], self.config_space.high_lims[0])
        ax.set_ylim(self.config_space.low_lims[1], self.config_space.high_lims[1])

        for obs in self.config_space.obstacles:
            xc, yc, r = obs
            circle = plt.Circle((xc, yc), r, color='black')
            ax.add_artist(circle)

        for path in self.graph.get_edge_paths():
            xs = path.positions[:, 0]
            ys = path.positions[:, 1]
            ax.plot(xs, ys, color='orange')

        if self.plan:
            plan_x = self.plan.positions[:, 0]
            plan_y = self.plan.positions[:, 1]
            ax.plot(plan_x, plan_y, color='green')

        plt.show()

def main():
    """Use this function if you'd like to test without ROS.

    If you're testing at home without ROS, you might want
    to get rid of the rospy.is_shutdown check in the main 
    planner loop (and the corresponding rospy import).
    """
    start = np.array([1, 1, 0, 0]) 
    goal = np.array([9, 9, 0, 0])
    xy_low = [0, 0]
    xy_high = [10, 10]
    u1_max = 2
    u2_max = 3
    obstacles_hard = [[6, 3.5, 1.5], [3.5, 6.5, 1], [8,7,0.5],[6,6,0.8], [1,7,0.7], [2,3,1.5]]
    obstacles_easy = [[6, 3.5, 1.5], [3.5, 6.5, 1]]

    config = UnicycleConfigurationSpace( xy_low,
                                        xy_high,
                                        [-u1_max, -u2_max],
                                        [u1_max, u2_max],
                                        obstacles_easy,
                                        0.15,start,goal)

    planner = RRTPlanner(config, max_iter=10000, expand_dist=3)
    plan = planner.plan_to_pose(start, goal)
    planner.plot_execution()

if __name__ == '__main__':
    main()
