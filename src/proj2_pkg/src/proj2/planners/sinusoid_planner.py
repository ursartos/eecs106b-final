#!/usr/bin/env python
"""
Starter code for EE106B Turtlebot Lab
Author: Valmik Prabhu, Chris Correa
Adapted for Spring 2020 by Amay Saxena
"""
import numpy as np
from scipy.integrate import quad
import sys
from copy import copy
import matplotlib.pyplot as plt
from .configuration_space import Plan, BicycleConfigurationSpace
import math
#binary search
from bisect import bisect_left

class SinusoidPlanner():
    def __init__(self, config_space):
        """
        Turtlebot planner that uses sequential sinusoids to steer to a goal pose.

        config_space should be a BicycleConfigurationSpace object.
        Parameters
        ----------
        l : float
            length of car
        """
        self.config_space = config_space
        self.l = config_space.robot_length
        self.max_phi = config_space.high_lims[3]
        self.max_u1 = config_space.input_high_lims[0]
        self.max_u2 = config_space.input_high_lims[1]

    def plan_to_pose(self, start_state, goal_state, dt = 0.01, delta_t=2):
        """
        Plans to a specific pose in (x,y,theta,phi) coordinates.  You 
        may or may not have to convert the state to a v state with state2v()
        You may want to plan each component separately
        so that you can reset phi in case there's drift in phi.

        You will need to edit some or all of this function to take care of
        configuration

        Parameters
        ----------
        start_state: numpy.ndarray of shape (4,) [x, y, theta, phi]
        goal_state: numpy.ndarray of shape (4,) [x, y, theta, phi]
        dt : float
            how many seconds between trajectory timesteps
        delta_t : float
            how many seconds each trajectory segment should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """

        print("======= Planning with SinusoidPlanner =======")

        self.plan = None
        # This bit hasn't been exhaustively tested, so you might hit a singularity anyways
        x_s, y_s, theta_s, phi_s = start_state
        x_g, y_g, theta_g, phi_g = goal_state
        max_abs_angle = max(abs(theta_g), abs(theta_s))
        min_abs_angle = min(abs(theta_g), abs(theta_s))
        if (max_abs_angle > np.pi/2) and (min_abs_angle < np.pi/2):
            #raise ValueError("You'll cause a singularity here. You should add something to this function to fix it")
            goal_state1 = [x_g, y_g, theta_g/3, phi_g]
            goal_state2 = [x_g, y_g, theta_g*2/3, phi_g]
            goal_state3 = [x_g, y_g, theta_g, phi_g]
            delta_t = delta_t/3

            print("Goal states:", goal_state1, goal_state2, goal_state3)

            self.phi_dist = min(
            abs(phi_g - self.max_phi),
            abs(phi_g + self.max_phi)
            )

            x_path1 =    self.steer_x(start_state, goal_state1, dt=dt, delta_t=delta_t)
            phi_path1 =  self.steer_phi(x_path1.end_position(),goal_state1,dt=dt, delta_t=delta_t)
            alpha_path1 = self.steer_alpha(phi_path1.end_position(), goal_state1, dt=dt, delta_t=delta_t)
            y_path1 =  self.steer_y(alpha_path1.end_position(),goal_state1,dt=dt, delta_t=delta_t)

            y_path2 =  self.steer_y_alt(y_path1.end_position(), goal_state2,dt=dt, delta_t=delta_t) 
            phi_path2 =  self.steer_phi_alt(y_path2.end_position(),goal_state2,dt=dt, delta_t=delta_t)
            alpha_path2 = self.steer_alpha_alt(phi_path2.end_position(), goal_state2, dt=dt, delta_t=delta_t)
            x_path2 = self.steer_x_alt(alpha_path2.end_position(), goal_state2, dt=dt, delta_t=delta_t)

            x_path3 =    self.steer_x(x_path2.end_position(), goal_state3, dt=dt, delta_t=delta_t)
            phi_path3 =  self.steer_phi(x_path3.end_position(),goal_state3,dt=dt, delta_t=delta_t)
            alpha_path3 = self.steer_alpha(phi_path3.end_position(), goal_state3, dt=dt, delta_t=delta_t)
            y_path3 =  self.steer_y(alpha_path3.end_position(),goal_state3,dt=dt, delta_t=delta_t) 


            self.plan = Plan.chain_paths(x_path1, phi_path1, alpha_path1, y_path1, y_path2, phi_path2, alpha_path2, x_path2,
                                        x_path3, phi_path3, alpha_path3, y_path3)
            return self.plan

        if abs(phi_s) > self.max_phi or abs(phi_g) > self.max_phi:
            raise ValueError("Either your start state or goal state exceeds steering angle bounds")

        # We can only change phi up to some threshold
        self.phi_dist = min(
            abs(phi_g - self.max_phi),
            abs(phi_g + self.max_phi)
        )

        x_path =        self.steer_x(
                            start_state, 
                            goal_state, 
                            dt=dt, 
                            delta_t=delta_t
                        )

        phi_path =      self.steer_phi(
                            x_path.end_position(), 
                            goal_state,  
                            dt=dt, 
                            delta_t=delta_t
                        )
        alpha_path =    self.steer_alpha(
                            phi_path.end_position(), 
                            goal_state, 
                            dt=dt, 
                            delta_t=delta_t
                        )
        y_path =        self.steer_y(
                            alpha_path.end_position(), 
                            goal_state,
                            dt=dt,
                            delta_t=delta_t
                        )     

        self.plan = Plan.chain_paths(x_path, phi_path, alpha_path, y_path)
        return self.plan

    def plot_execution(self):
        """
        Creates a plot of the planned path in the environment. Assumes that the 
        environment of the robot is in the x-y plane, and that the first two
        components in the state space are x and y position. Also assumes 
        plan_to_pose has been called on this instance already, so that self.graph
        is populated. If planning was successful, then self.plan will be populated 
        and it will be plotted as well.
        """
        ax = plt.subplot(1, 1, 1)

        if self.plan:
            plan_x = self.plan.positions[:, 0]
            plan_y = self.plan.positions[:, 1]
            ax.plot(plan_x, plan_y, color='green')

        plt.show()

    def steer_x(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2):
        """
        Create a Plan to move the turtlebot in the x direction

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        start_state_v = self.state2v(start_state)
        goal_state_v = self.state2v(goal_state)
        delta_x = goal_state_v[0] - start_state_v[0]

        v1 = delta_x/delta_t
        v2 = 0

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1, v2])
            t = t + dt
        return self.v_path_to_u_path(path, start_state, dt)

    def steer_phi(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2):
        """
        Create a trajectory to move the turtlebot in the phi direction

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        goal_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        start_state_v = self.state2v(start_state)
        goal_state_v = self.state2v(goal_state)
        delta_phi = goal_state_v[1] - start_state_v[1]

        v1 = 0
        v2 = delta_phi/delta_t

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1, v2])
            t = t + dt
        return self.v_path_to_u_path(path, start_state, dt)

    def steer_alpha(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2):
        """
        Create a trajectory to move the turtlebot in the alpha direction.  
        Remember dot{alpha} = f(phi(t))*u_1(t) = f(frac{a_2}{omega}*sin(omega*t))*a_1*sin(omega*t)
        also, f(phi) = frac{1}{l}tan(phi)
        See the doc for more math details

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        goal_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """

        start_state_v = self.state2v(start_state)
        goal_state_v = self.state2v(goal_state)
        delta_alpha = goal_state_v[2] - start_state_v[2]

        omega = 2*np.pi / delta_t

        a2 = min(1, self.phi_dist*omega)
        f = lambda phi: (1/self.l)*np.tan(phi) # This is from the car model
        phi_fn = lambda t: (a2/omega)*np.sin(omega*t) + start_state_v[1]
        integrand = lambda t: f(phi_fn(t))*np.sin(omega*t) # The integrand to find beta
        beta1 = (omega/np.pi) * quad(integrand, 0, delta_t)[0]

        a1 = (delta_alpha*omega)/(np.pi*beta1)
              
        v1 = lambda t: a1*np.sin(omega*(t))
        v2 = lambda t: a2*np.cos(omega*(t))

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1(t-t0), v2(t-t0)])
            t = t + dt
        return self.v_path_to_u_path(path, start_state, dt)


    def steer_y(self, start_state, goal_state, t0 = 0.0, dt = 0.01, delta_t = 2.0):
        """
        Create a trajectory to move the turtlebot in the y direction. 
        Remember, dot{y} = g(alpha(t))*v1 = frac{alpha(t)}{sqrt{1-alpha(t)^2}}*a_1*sin(omega*t)
        See the doc for more math details

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        goal_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        start_state_v = self.state2v(start_state)
        goal_state_v = self.state2v(goal_state)
    
        delta_y = goal_state_v[3] - start_state_v[3]
        omega = 2*np.pi / delta_t

        a2 = min(1, self.phi_dist*omega) #guessing value

        a1_list = np.linspace(0,3.0/a2,1000) #list of a1, unsure of reasonable values
        diff_y_list = [] #list of y(2*pi/omega) - y(0) values for corresponding a1
        for a1 in a1_list:
            #Inner integral
            f = lambda phi: (1/self.l)*np.tan(phi) # This is from the car model
            phi_fn = lambda t1: a2/(2.0*omega)*np.sin(2.0*omega*t1) + start_state_v[1] 
            integrand_1 = lambda t2: f(phi_fn(t2))*a1*np.sin(omega*t2)
            inner_integral = lambda t3: quad(integrand_1, t0, t3)[0] + start_state_v[2]

            #Outer integral
            g = lambda alpha: alpha/np.sqrt(1.0-alpha**2) # This is from the car model

            integrand_2 = lambda t4: g(inner_integral(t4))*np.sin(omega*t4)
            beta1 = (omega/np.pi) * quad(integrand_2, t0, 2.0*np.pi/omega)[0]
            diff_y = np.pi*a1*beta1 / omega
            diff_y_list.append(diff_y)

        def binarySearch(data, val):
            lo, hi = 0, len(data) - 1
            best_ind = lo
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if data[mid] < val:
                    lo = mid + 1
                elif data[mid] > val:
                    hi = mid - 1
                else:
                    best_ind = mid
                    break
                # check if data[mid] is closer to val than data[best_ind] 
                if abs(data[mid] - val) < abs(data[best_ind] - val):
                    best_ind = mid
            return best_ind

        #Binary search
        idx = binarySearch(diff_y_list, delta_y)
        print("compare", diff_y_list[idx], delta_y, idx)
        a1 = a1_list[idx]
        
        v1 = lambda t: a1*np.sin(omega*(t))
        v2 = lambda t: a2*np.cos(2*omega*(t))


        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1(t-t0), v2(t-t0)])
            t = t + dt
        return self.v_path_to_u_path(path, start_state, dt)

        #Todo: Implement constraints

        #Todo: Change between alternating and cannoncal model for different theta
        #use these: max_abs_angle and min_abs_angle 

        #run main.py: main.py -planner=sin -x=2 -y=1.3 -theta=0.7 -phi=0
        # or src/proj2_pkg/scripts/main.py -planner=sin -x=2 -y=1.3 -theta=0.7 -phi=0

    def steer_y_alt(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2):
        """
        Create a Plan to move the turtlebot in the x direction

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        start_state_v = self.state2v_alt(start_state)
        goal_state_v = self.state2v_alt(goal_state)
        delta_x = goal_state_v[0] - start_state_v[0]

        v1 = delta_x/delta_t
        v2 = 0

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1, v2])
            t = t + dt
        return self.v_path_to_u_path_alt(path, start_state, dt)

    def steer_phi_alt(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2):
        """
        Create a trajectory to move the turtlebot in the phi direction

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        goal_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        start_state_v = self.state2v_alt(start_state)
        goal_state_v = self.state2v_alt(goal_state)
        delta_phi = goal_state_v[1] - start_state_v[1]

        v1 = 0
        v2 = delta_phi/delta_t

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1, v2])
            t = t + dt
        return self.v_path_to_u_path_alt(path, start_state, dt)

    def steer_alpha_alt(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2):
        """
        Create a trajectory to move the turtlebot in the alpha direction.  
        Remember dot{alpha} = f(phi(t))*u_1(t) = f(frac{a_2}{omega}*sin(omega*t))*a_1*sin(omega*t)
        also, f(phi) = frac{1}{l}tan(phi)
        See the doc for more math details

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        goal_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """

        start_state_v = self.state2v_alt(start_state)
        goal_state_v = self.state2v_alt(goal_state)
        delta_alpha = goal_state_v[2] - start_state_v[2]

        omega = 2*np.pi / delta_t

        a2 = min(1, self.phi_dist*omega)
        f = lambda phi: -(1/self.l)*np.tan(phi) # This is from the car model
        phi_fn = lambda t: (a2/omega)*np.sin(omega*t) + start_state_v[1]
        integrand = lambda t: f(phi_fn(t))*np.sin(omega*t) # The integrand to find beta
        beta1 = (omega/np.pi) * quad(integrand, 0, delta_t)[0]

        a1 = (delta_alpha*omega)/(np.pi*beta1)
              
        v1 = lambda t: a1*np.sin(omega*(t))
        v2 = lambda t: a2*np.cos(omega*(t))

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1(t-t0), v2(t-t0)])
            t = t + dt
        return self.v_path_to_u_path_alt(path, start_state, dt)


    def steer_x_alt(self, start_state, goal_state, t0 = 0.0, dt = 0.01, delta_t = 2.0):
        """
        Create a trajectory to move the turtlebot in the y direction. 
        Remember, dot{y} = g(alpha(t))*v1 = frac{alpha(t)}{sqrt{1-alpha(t)^2}}*a_1*sin(omega*t)
        See the doc for more math details

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        goal_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        start_state_v = self.state2v_alt(start_state)
        goal_state_v = self.state2v_alt(goal_state)
    
        delta_y = goal_state_v[3] - start_state_v[3]
        omega = 2*np.pi / delta_t

        a2 = min(1, self.phi_dist*omega) #guessing value

        a1_list = np.linspace(0,3.0/a2,1000) #list of a1, unsure of reasonable values
        diff_y_list = [] #list of y(2*pi/omega) - y(0) values for corresponding a1
        for a1 in a1_list:
            #Inner integral
            f = lambda phi: -(1/self.l)*np.tan(phi) # This is from the car model
            phi_fn = lambda t1: a2/(2.0*omega)*np.sin(2.0*omega*t1) + start_state_v[1] 
            integrand_1 = lambda t2: f(phi_fn(t2))*a1*np.sin(omega*t2)
            inner_integral = lambda t3: quad(integrand_1, t0, t3)[0] + start_state_v[2]

            #Outer integral
            g = lambda alpha: alpha/np.sqrt(1.0-alpha**2) # This is from the car model

            integrand_2 = lambda t4: g(inner_integral(t4))*np.sin(omega*t4)
            beta1 = (omega/np.pi) * quad(integrand_2, t0, 2.0*np.pi/omega)[0]
            diff_y = np.pi*a1*beta1 / omega
            diff_y_list.append(diff_y)

        def binarySearch(data, val):
            lo, hi = 0, len(data) - 1
            best_ind = lo
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if data[mid] < val:
                    lo = mid + 1
                elif data[mid] > val:
                    hi = mid - 1
                else:
                    best_ind = mid
                    break
                # check if data[mid] is closer to val than data[best_ind] 
                if abs(data[mid] - val) < abs(data[best_ind] - val):
                    best_ind = mid
            return best_ind

        #Binary search
        idx = binarySearch(diff_y_list, delta_y)
        print("compare", diff_y_list[idx], delta_y, idx)
        a1 = a1_list[idx]
        
        v1 = lambda t: a1*np.sin(omega*(t))
        v2 = lambda t: a2*np.cos(2*omega*(t))


        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1(t-t0), v2(t-t0)])
            t = t + dt
        return self.v_path_to_u_path_alt(path, start_state, dt)

        #Todo: Implement constraints

        #Todo: Change between alternating and cannoncal model for different theta
        #use these: max_abs_angle and min_abs_angle 

        #run main.py: main.py -planner=sin -x=2 -y=1.3 -theta=0.7 -phi=0
        # or src/proj2_pkg/scripts/main.py -planner=sin -x=2 -y=1.3 -theta=0.7 -phi=0




    def state2v(self, state):
        """
        Takes a state in (x,y,theta,phi) coordinates and returns a state of (x,phi,alpha,y)

        Parameters
        ----------
        state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            some state

        Returns
        -------
        4x1 :obj:`numpy.ndarray` 
            x, phi, alpha, y
        """
        x, y, theta, phi = state
        return np.array([x, phi, np.sin(theta), y])


    def state2v_alt(self, state):
        """
        Takes a state in (x,y,theta,phi) coordinates and returns a state of (y,phi,alpha,x)

        Parameters
        ----------
        state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            some state

        Returns
        -------
        4x1 :obj:`numpy.ndarray` 
            y, phi, alpha, x
        """
        x, y, theta, phi = state
        return np.array([y, phi, np.cos(theta), x])

    def v_path_to_u_path(self, path, start_state, dt):
        """
        convert a trajectory in v commands to u commands

        Parameters
        ----------
        path : :obj:`list` of (float, float, float)
            list of (time, v1, v2) commands
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            starting state of this trajectory
        dt : float
            how many seconds between timesteps in the trajectory

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        def v2cmd(v1, v2, state):
            u1 = v1/np.cos(state[2])
            u2 = v2
            return [u1, u2]

        curr_state = start_state
        positions = []
        times = []
        open_loop_inputs = []
        for i, (t, v1, v2) in enumerate(path):
            cmd_u = v2cmd(v1, v2, curr_state)
            positions.append(curr_state)
            open_loop_inputs.append(cmd_u)
            times.append(t)

            x, y, theta, phi = curr_state
            linear_velocity, steering_rate = cmd_u
            curr_state = [
                x     + np.cos(theta)               * linear_velocity*dt,
                y     + np.sin(theta)               * linear_velocity*dt,
                theta + np.tan(phi) / float(self.l) * linear_velocity*dt,
                phi   + steering_rate*dt
            ]

        return Plan(np.array(times), np.array(positions), np.array(open_loop_inputs), dt=dt)



    def v_path_to_u_path_alt(self, path, start_state, dt):
        """
        convert a trajectory in v commands to u commands

        Parameters
        ----------
        path : :obj:`list` of (float, float, float)
            list of (time, v1, v2) commands
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            starting state of this trajectory
        dt : float
            how many seconds between timesteps in the trajectory

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        def v2cmd(v1, v2, state):
            u1 = v1/np.sin(state[2])
            u2 = v2
            return [u1, u2]

        curr_state = start_state
        positions = []
        times = []
        open_loop_inputs = []
        for i, (t, v1, v2) in enumerate(path):
            cmd_u = v2cmd(v1, v2, curr_state)
            positions.append(curr_state)
            open_loop_inputs.append(cmd_u)
            times.append(t)

            x, y, theta, phi = curr_state
            linear_velocity, steering_rate = cmd_u
            curr_state = [
                x     + np.cos(theta)               * linear_velocity*dt,
                y     + np.sin(theta)               * linear_velocity*dt,
                theta + np.tan(phi) / float(self.l) * linear_velocity*dt,
                phi   + steering_rate*dt
            ]

        return Plan(np.array(times), np.array(positions), np.array(open_loop_inputs), dt=dt)

def main():
    """Use this function if you'd like to test without ROS.
    """
    start = np.array([1, 1, 0, 0]) 
    goal = np.array([2, 1.3, 0.7, 0])
    xy_low = [0, 0]
    xy_high = [5, 5]
    phi_max = 0.6
    u1_max = 2
    u2_max = 3
    obstacles = []

    config = BicycleConfigurationSpace( xy_low + [-1000, -phi_max],
                                        xy_high + [1000, phi_max],
                                        [-u1_max, -u2_max],
                                        [u1_max, u2_max],
                                        obstacles,
                                        0.15)

    planner = SinusoidPlanner(config)
    plan = planner.plan_to_pose(start, goal, 0.01, 2.0)
    planner.plot_execution()





if __name__ == '__main__':
    main()
