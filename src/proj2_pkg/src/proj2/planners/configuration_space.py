#!/usr/bin/env python

"""
Starter code for EECS C106B Spring 2020 Project 2.
Author: Amay Saxena
"""
import time
from ast import mod
from distutils.command.build_scripts import first_line_re
from operator import truediv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from contextlib import contextmanager
import random

#from sympy import primitive

class Plan(object):
    """Data structure to represent a motion plan. Stores plans in the form of
    three arrays of the same length: times, positions, and open_loop_inputs.

    The following invariants are assumed:
        - at time times[i] the plan prescribes that we be in position
          positions[i] and perform input open_loop_inputs[i].
        - times starts at zero. Each plan is meant to represent the motion
          from one point to another over a time interval starting at 
          time zero. If you wish to append together multiple paths
          c1 -> c2 -> c3 -> ... -> cn, you should use the chain_paths
          method.
    """

    def __init__(self, times, target_positions, open_loop_inputs, dt=0.01):
        self.dt = dt
        self.times = times
        self.positions = target_positions
        self.open_loop_inputs = open_loop_inputs

    def __iter__(self):
        # I have to do this in an ugly way because python2 sucks and
        # I hate it.
        for t, p, c in zip(self.times, self.positions, self.open_loop_inputs):
            yield t, p, c

    def __len__(self):
        return len(self.times)

    def get(self, t):
        """Returns the desired position and open loop input at time t.
        """
        index = int(np.sum(self.times <= t))
        index = index - 1 if index else 0
        return self.positions[index], self.open_loop_inputs[index]

    def end_position(self):
        return self.positions[-1]

    def start_position(self):
        return self.positions[0]
    
    #def print(self):
        #print(f"primitive.times: {self.times[0]} to {self.times[-1]}")
        #print(f"primitive.positions: {self.positions[0]} to {self.positions[-1]}")
        #print(f"primitive.open_loop_inputs: {self.open_loop_inputs[0]}")

    def get_prefix(self, until_time):
        """Returns a new plan that is a prefix of this plan up until the
        time until_time.
        """
        times = self.times[self.times <= until_time]
        positions = self.positions[self.times <= until_time]
        #print(f"self.open_loop_inputs {self.open_loop_inputs}")
        self.open_loop_inputs = np.array(self.open_loop_inputs)
        open_loop_inputs = self.open_loop_inputs[self.times <= until_time]
        return Plan(times, positions, open_loop_inputs)

    @classmethod
    def chain_paths(self, *paths):
        """Chain together any number of plans into a single plan.
        """
        def chain_two_paths(path1, path2):
            """Chains together two plans to create a single plan. Requires
            that path1 ends at the same configuration that path2 begins at.
            Also requires that both paths have the same discretization time
            step dt.
            """
            if path1 != None:
                for i in range(len(path1.positions)):
                    path1.positions[i][2] = path1.positions[i][2] % (2*np.math.pi)
            if path2 != None:
                for j in range(len(path2.positions)):
                    path2.positions[j][2] = path2.positions[j][2] % (2*np.math.pi)

            if not path1 and not path2:
                return None
            elif not path1:
                return path2
            elif not path2:
                return path1

            

            

            assert path1.dt == path2.dt, "Cannot append paths with different time deltas."
            #print(f"path positions: {path1.end_position(),path2.start_position()}")
            assert np.allclose(path1.end_position(), path2.start_position()), "Cannot append paths with inconsistent start and end positions."
            times = np.concatenate((path1.times, path1.times[-1] + path2.times[1:]), axis=0)
            positions = np.concatenate((path1.positions, path2.positions[1:]), axis=0)
            open_loop_inputs = np.concatenate((path1.open_loop_inputs, path2.open_loop_inputs[1:]), axis=0)
            dt = path1.dt
            return Plan(times, positions, open_loop_inputs, dt=dt)
        chained_path = None
        for path in paths:
            chained_path = chain_two_paths(chained_path, path)
        return chained_path

@contextmanager
def expanded_obstacles(obstacle_list, delta):
    """Context manager that edits obstacle list to increase the radius of
    all obstacles by delta.
    
    Assumes obstacles are circles in the x-y plane and are given as lists
    of [x, y, r] specifying the center and radius of the obstacle. So
    obstacle_list is a list of [x, y, r] lists.

    Note we want the obstacles to be lists instead of tuples since tuples
    are immutable and we would be unable to change the radii.

    Usage:
        with expanded_obstacles(obstacle_list, 0.1):
            # do things with expanded obstacle_list. While inside this with 
            # block, the radius of each element of obstacle_list has been
            # expanded by 0.1 meters.
        # once we're out of the with block, obstacle_list will be
        # back to normalpositions[i]
    """
    for obs in obstacle_list:
        obs[2] += delta
    yield obstacle_list
    for obs in obstacle_list:
        obs[2] -= delta

class ConfigurationSpace(object):
    """ An abstract class for a Configuration Space. 
    
        DO NOT FILL IN THIS CLASS

        Instead, fill in the BicycleConfigurationSpace at the bottom of the
        file which inherits from this class.
    """

    def __init__(self, dim, low_lims, high_lims, obstacles, dt=0.01, constraints=None):
        """
        Parameters
        ----------
        dim: dimension of the state space: number of state variables.
        low_lims: the lower bounds of the state variables. Should be an
                iterable of length dim.
        high_lims: the higher bounds of the state variables. Should be an
                iterable of length dim.
        obstacles: A list of obstacles. This could be in any representation
            we choose, based on the application. In this project, for the bicycle
            model, we assume each obstacle is a circle in x, y space, and then
            obstacles is a list of [x, y, r] lists specifying the center and 
            radius of each obstacle.
        dt: The discretization timestep our local planner should use when constructing
            plans.
        """
        self.dim = dim
        self.low_lims = np.array(low_lims)
        self.high_lims = np.array(high_lims)
        self.obstacles = obstacles
        self.constraints = constraints
        self.dt = dt

    def distance(self, c1, c2):
        """
            Implements the chosen metric for this configuration space.
            This method should be implemented whenever this ConfigurationSpace
            is subclassed.

            Returns the distance between configurations c1 and c2 according to
            the chosen metric.
        """
        pass

    def sample_config(self, *args):
        """
            Samples a new configuration from this C-Space according to the
            chosen probability measure.
            This method should be implemented whenever this ConfigurationSpace
            is subclassed.

            Returns a new configuration sampled at random from the configuration
            space.
        """
        pass

    def check_collision(self, c):
        """
            Checks to see if the specified configuration c is in collision with
            any obstacles.
            This method should be implemented whenever this ConfigurationSpace
            is subclassed.
        """
        pass

    def check_path_collision(self, path):
        """
            Checks to see if a specified path through the configuration space is 
            in collision with any obstacles.
            This method should be implemented whenever this ConfigurationSpace
            is subclassed.
        """
        pass

    def local_plan(self, c1, c2):
        """
            Constructs a plan from configuration c1 to c2.

            This is the local planning step in RRT. This should be where you extend
            the trajectory of the robot a little bit starting from c1. This may not
            constitute finding a complete plan from c1 to c2. Remember that we only
            care about moving in some direction while respecting the kinemtics of
            the robot. You may perform this step by picking a number of motion
            primitives, and then returning the primitive that brings you closest
            to c2.
        """
        pass

    def nearest_config_to(self, config_list, config):
        """
            Finds the configuration from config_list that is closest to config.
        """
        return min(config_list, key=lambda c: self.distance(c, config))

class FreeEuclideanSpace(ConfigurationSpace):
    """
        Example implementation of a configuration space. This class implements
        a configuration space representing free n dimensional euclidean space.
    """

    def __init__(self, dim, low_lims, high_lims, sec_per_meter=4):
        super(FreeEuclideanSpace, self).__init__(dim, low_lims, high_lims, [])
        self.sec_per_meter = sec_per_meter

    def distance(self, c1, c2):
        """
        c1 and c2 should by numpy.ndarrays of size (dim, 1) or (1, dim) or (dim,).
        """
        return np.linalg.norm(c1 - c2)

    def sample_config(self, *args):
        return np.random.uniform(self.low_lims, self.high_lims).reshape((self.dim,))

    def check_collision(self, c):
        return False

    def check_path_collision(self, path):
        return False

    def local_plan(self, c1, c2):
        v = c2 - c1
        dist = np.linalg.norm(c1 - c2)
        total_time = dist * self.sec_per_meter
        vel = v / total_time
        p = lambda t: (1 - (t / total_time)) * c1 + (t / total_time) * c2
        times = np.arange(0, total_time, self.dt)
        positions = p(times[:, None])
        velocities = np.tile(vel, (positions.shape[0], 1))
        plan = Plan(times, positions, velocities, dt=self.dt)
        return plan

class UnicycleConfigurationSpace(ConfigurationSpace):
    """
        The configuration space for a Bicycle modeled robot
        Obstacles should be tuples (x, y, r), representing circles of 
        radius r centered at (x, y)
        We assume that the robot is circular and has radius equal to robot_radius
        The state of the robot is defined as (x, y, theta, phi).
    """
    def __init__(self, low_lims, high_lims, input_low_lims, input_high_lims, obstacles, robot_radius,start,goal,terrains=[]):
        dim = 4
        super(UnicycleConfigurationSpace, self).__init__(dim,low_lims,high_lims,obstacles, terrains=None)
        self.robot_radius = robot_radius
        self.robot_length = 0.3
        self.input_low_lims = input_low_lims
        self.input_high_lims = input_high_lims
        self.start = start
        self.goal = goal
        self.linear_distance = np.sqrt(((self.start[0]-self.goal[0]) ** 2) + ((self.start[1]-self.goal[1]) ** 2))
        self.terrains = terrains
        
    
    def distance(self, c1, c2, alpha = 0.05,beta = 1, multiplier = 8):
        """
        c1 and c2 should be numpy.ndarrays of size (4,)
        """
        linear_distance = np.sqrt(((c1[0]-c2[0]) ** 2) + ((c1[1]-c2[1]) ** 2))
        angle_a = c1[2]-c2[2]
        #angle_a = (angle_a + np.math.pi) %np.math.pi - np.math.pi
        angle_a = np.absolute(angle_a)
        angle_b = (np.math.pi*2) - angle_a
        angle_b = np.absolute(angle_b)
        angular_distance = min(angle_a,angle_b)
        #angular_distance = np.absolute(angle_a)
        #print(c1,c2)
        #print("1:",abs(c1[2]-c2[2]))
        #print("2",(np.math.pi*2) - abs(c1[2]-c2[2]))
        #print(f"in the distance c1 = {c1} c2 = {c2}")
        #angular_distance = np.min(int(abs(c1[2]-c2[2])), int((np.math.pi*2) - abs(c1[2]-c2[2])))
        return (alpha * multiplier * linear_distance) + (beta * multiplier * angular_distance)

    def sample_config(self, goal, goal_sample_prob = 0.1, inner_radius = 0.3, inner_sample_prob = 0.25, outer_radius = 0.4, outer_sample_prob = 0.2, counter = 0):
        """
        Pick a random configuration from within our state boundaries.

        You can pass in any number of additional optional arguments if you
        would like to implement custom sampling heuristics. By default, the
        RRT implementation passes in the goal as an additional argument,
        which can be used to implement a goal-biasing heuristic.
        """
        

        def sample_point_within_radius_of_goal(radius_as_decimal):

            #radius = (self.high_lims[0] - self.low_lims[0]) * radius_as_decimal
            radius = self.linear_distance * radius_as_decimal


            goal_lower_x = max(self.low_lims[0],goal[0] - radius)
            goal_upper_x = min(self.high_lims[0],goal[0] + radius)

            goal_lower_y = max(self.low_lims[1],goal[1] - radius)
            goal_upper_y = min(self.high_lims[1],goal[1] + radius)

            point_x = random.uniform(goal_lower_x,goal_upper_x)
            point_y = random.uniform(goal_lower_y,goal_upper_y)
            point = [point_x,point_y]
            return point

        def sample_angle_with_respect_to_goal(point):
            '''CHANGE THIS LATER, FOR NOW WE WILL JUST SAMPLE A RANDOM THETA'''
            angle = random.uniform(0,np.math.pi * 2)
            return [angle]

        sleep = 0.00
        if counter % (1/goal_sample_prob) == 0:
            #print("sampling goal")
            time.sleep(sleep)
            return goal
        elif counter % (1/inner_sample_prob) == 0:
            xy = sample_point_within_radius_of_goal(inner_radius)
            angle = sample_angle_with_respect_to_goal(xy)
            #print("sampling inner_radius")
            #print(f"point = {xy}")
            time.sleep(sleep)
            return np.array(xy + angle)
        elif counter % (1/outer_sample_prob) == 0:
            xy = sample_point_within_radius_of_goal(outer_radius)
            angle = sample_angle_with_respect_to_goal(xy)
            #print("sampling outer_radius")
            #print(f"point = {xy}")
            time.sleep(sleep)
            return np.array(xy + angle)
        else:
            xy = sample_point_within_radius_of_goal(1)
            angle = sample_angle_with_respect_to_goal(xy)
            #print("sampling rest")
            time.sleep(sleep)
            #print(f"point = {xy}")
            return np.array(xy + angle)
        
        

    def check_if_point_in_circle(self,x,y,c_x,c_y,c_r):
        if ((x - c_x)**2 + (y - c_y)**2 <= (c_r**2)):
            return True
        else:
            return False

    def check_collision(self, c):
        """
        Returns true if a configuration c is in collision
        c should be a numpy.ndarray of size (4,)
        """
        flag = False

        with expanded_obstacles(self.obstacles, self.robot_radius):
            for o in self.obstacles:
                #print("c:",c,"o",o)
                if self.check_if_point_in_circle(c[0],c[1],o[0],o[1],o[2]):
                    flag = True
                    break

        return flag


    

    def check_path_collision(self, path, looseness = 1):
        """
        Returns true if the input path is in collision. The path
        is given as a Plan object. See configuration_space.py
        for details on the Plan interface.

        You should also ensure that the path does not exceed any state bounds,
        and the open loop inputs don't exceed input bounds.
        """
        
        flag = False
        i = 0
        while i < len(path.times):
            #print(f"path.open_loop_inputs[i]={path.open_loop_inputs[i]} self.input_low_lims = {self.input_low_lims}")
            path.positions[i][2] = path.positions[i][2] + np.math.pi*2
            if self.check_collision(path.positions[i]):
                flag = True
                break
            # elif ((path.positions[i][2] < 0) or (path.positions[i][2] > np.math.pi *2)):
            #     print(i)
            #     print("path invalid on theta")
            #     flag = True
            #     break
            elif ((path.open_loop_inputs[i][0] < self.input_low_lims[0]) or (path.open_loop_inputs[i][1] < self.input_low_lims[1]) or (path.open_loop_inputs[i][0] > self.input_high_lims[0]) or (path.open_loop_inputs[i][1] > self.input_high_lims[1])):
                print("path invalid on inputs")
                flag = True
                break
            i += looseness

        return flag
        

    def local_plan(self, c1, c2, dt=0.01,timesteps = 50):
        """
        Constructs a local plan from c1 to c2. Usually, you want to
        just come up with any plan without worrying about obstacles,
        because the algorithm checks to see if the path is in collision,
        in which case it is discarded.

        However, in the case of the nonholonomic bicycle model, it will
        be very difficult for you to come up with a complete plan from c1
        to c2. Instead, you should choose a set of "motion-primitives", and
        then simply return whichever motion primitive brings you closest to c2.

        A motion primitive is just some small, local motion, that we can perform
        starting at c1. If we keep a set of these, we can choose whichever one
        brings us closest to c2.

        Keep in mind that choosing this set of motion primitives is tricky.
        Every plan we come up with will just be a bunch of these motion primitives
        chained together, so in order to get a complete motion planner, you need to 
        ensure that your set of motion primitives is such that you can get from any
        point to any other point using those motions.

        For example, in cartpositions[0] = start Then, this local planner would just amount to picking 
        the values of a1, a2, a3 that bring us closest to c2.

        You should spend some time thinking about what motion primitives would
        be good to use for a bicycle model robot. What kinds of motions are at
        our disposal?

        This should return a cofiguration_space.Plan object.

        """

        

        def construct_primitive(start,timesteps,u1,u2):
            1/0
            print("not unicycled")
            #print("duration:", duration )

            length = int(timesteps)
            #print("lenggth ",length)
            #print(int(duration))
            times = np.linspace(0,length*dt,int(length))
            #print(length)
            #print(np.array([u1,u2]))
            open_loop_input = np.array([u1,
                                        u2])
            # open_loop_inputs_top = np.array([u1] * 25)
            # open_loop_inputs_bottom = np.array([u2] * 25)
            # open_loop_inputs = np.vstack((open_loop_inputs_top,open_loop_inputs_bottom))
            open_loop_inputs = [open_loop_input for i in range(0,length)]
            positions = np.array([start])
            while len(positions) != length:
                current = positions[len(positions)-1]
                first = np.array([np.cos(current[2]),
                                  np.sin(current[2]),
                                  (1/self.robot_length) * current[3]])

                second = np.array([0,0,0,1])
                q_dot = (first * u1) + (second * u2)
                new_pos = current + q_dot*dt
                new_pos[2] = (new_pos[2] + 2*np.math.pi)  % (2*np.math.pi)
                if new_pos[3] >= 0.6:
                    new_pos[3] = 0.59
                if new_pos[3] <= -0.6:
                    new_pos[3] = -0.59
                positions = np.vstack([positions, new_pos])
            #print(f"primitive: times = {times} positions = {positions} open_loop_inputs = {open_loop_inputs}")
            return Plan(times,positions,open_loop_inputs,dt)
        

        #primitives is a list of plans
        def construct_set_of_primitives_about_c1():
            primitives = []
            primitives.append(construct_primitive(c1,timesteps,1,0 ))
            primitives.append(construct_primitive(c1,timesteps,-1,0 ))
            primitives.append(construct_primitive(c1,timesteps,1,0.6 ))
            primitives.append(construct_primitive(c1,timesteps,1,-0.6 ))
            primitives.append(construct_primitive(c1,timesteps,-1,0.6 ))
            primitives.append(construct_primitive(c1,timesteps,-1,-0.6))
            primitives.append(construct_primitive(c1,timesteps,1,0.3 ))
            primitives.append(construct_primitive(c1,timesteps,1,-0.3 ))
            primitives.append(construct_primitive(c1,timesteps,-1,0.3 ))
            primitives.append(construct_primitive(c1,timesteps,-1,-0.3))
            primitives.append(construct_primitive(c1,timesteps,1,3 ))
            primitives.append(construct_primitive(c1,timesteps,1,-3 ))
            primitives.append(construct_primitive(c1,timesteps,-1,3 ))
            primitives.append(construct_primitive(c1,timesteps,-1,-3))
            return primitives


        def distance_from_point_to_primitive(plan,target):
            end_point = plan.end_position()
            distance = self.distance(end_point,target)
            return distance

        #print("got here")
        min = 10000
        #print("START")
        set = construct_set_of_primitives_about_c1()
        
        #print("END")
        best = set[0]
        for p in range(len(set)):
            #print(" ")
            #print(f"{p}th primitive:")
            #set[p].print()
            #print(f"p = {p} c2 = {c2}")
            distance = distance_from_point_to_primitive(set[p],c2)
            #print(f"the {p}th primitive had a distance of {distance}")
            #print(f"distance = {distance}")
            #print(f"min = {min} distance = {distance}")
            if distance <= min:
                
                min = distance
                best = set[p]
        #print("")
        #print(f"local planner is about to return this best path:")
        #best.print()
        return best
