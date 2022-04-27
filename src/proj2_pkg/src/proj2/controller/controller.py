#!/usr/bin/env python

"""
Starter code for EECS C106B Spring 2020 Project 2.
Author: Amay Saxena
"""
import numpy as np
import sys

import tf2_ros
import tf
from std_srvs.srv import Empty as EmptySrv
import rospy
from proj2_pkg.msg import UnicycleCommandMsg, UnicycleStateMsg
from proj2.planners import RRTPlanner, UnicycleConfigurationSpace # SinusoidPlanner
from kernel_reg import ParameterEstimatorKernel

TERRAIN_DIM = 1

class UnicycleModelController(object):
    def __init__(self):
        """
        Executes a plan made by the planner
        """
        self.pub = rospy.Publisher('/unicycle/cmd_vel', UnicycleCommandMsg, queue_size=10)
        self.sub = rospy.Subscriber('/unicycle/state', UnicycleStateMsg, self.subscribe)
        self.state = UnicycleStateMsg()
        self.buffer = []
        self.d_estimator = ParameterEstimatorKernel()
        self.k_estimator = ParameterEstimatorKernel()
        self.d = 1
        self.k = 1
        rospy.on_shutdown(self.shutdown)

    def current_pos_to_terrain(self, pos, terrains):
        terrain_vec = np.zeros(TERRAIN_DIM + 1)
        for i, terrain in enumerate(terrains):
            terrain_corners = terrain[0]
            if terrain_corners[0] <= pos[0] <= terrain_corners[1] and terrain_corners[2] <= pos[1] <= terrain_corners[3]:
                terrain_vec[i + 1] = 1
                return terrain_vec, terrain[1], terrain[2]
        terrain_vec[0] = 1
        return terrain_vec, 1, 1

    def mock_velocity(self, pos, commanded, terrains):
        d = 1
        k = 1
        for i, terrain in enumerate(terrains):
            terrain_corners = terrain[0]
            if terrain_corners[0] <= pos[0] <= terrain_corners[1] and terrain_corners[2] <= pos[1] <= terrain_corners[3]:
                d, k = terrain[1], terrain[2]
        return [commanded[0] * d, commanded[1] * k]

    def execute_plan(self, plan, terrains=[]):
        """
        Executes a plan made by the planner

        Parameters
        ----------
        plan : :obj: Plan. See configuration_space.Plan
        """
        if len(plan) == 0:
            return
        print("plan dt", plan.dt)
        rate = rospy.Rate(int(1 / plan.dt))
        start_t = rospy.Time.now()
        cur_state = self.state
        self.vel_cmd = 0
        cur_velocity = np.zeros(cur_state.shape)
        prev_t = start_t.to_sec()

        self.d_est = np.zeros((0,))
        self.k_est = np.zeros((0,))
        self.d_goal = np.zeros((0,))
        self.k_goal = np.zeros((0,))

        inputs_agg = []
        prev_state_change_time = prev_t

        sys_id_count = 0
        sys_id_period = 3
        while not rospy.is_shutdown():
            t = (rospy.Time.now() - start_t).to_sec()

            dt = plan.dt
            if t < plan.times[-1]:
                state, cmd = plan.get(t)
                next_state, next_cmd = plan.get(t+dt)
                prev_state, prev_cmd = plan.get(t-dt)
            elif t < plan.times[-1] + 0:
                cmd = cmd*0
            else:
                break

            current_terrain_vector = self.current_pos_to_terrain(state[:2], terrains)[0] # eventually this will be made into vision-based
            target_acceleration = ((next_state - state)/dt - (state - prev_state)/dt)/dt
            target_velocity = ((next_state - state)/dt)

            if np.linalg.norm(self.state - cur_state) > 0:
                cur_velocity = (self.state - cur_state)/(t-prev_state_change_time)
                sys_id_count += 1

                if sys_id_count % sys_id_period == 0:
                    self.buffer.append((cur_state, self.state, t-prev_state_change_time, np.mean(inputs_agg, axis=0), current_terrain_vector))
                    # print("visual feature", current_terrain_vector)
                    self.estimate_parameters_kernel(self.buffer)

                prev_state_change_time = t

            cur_state = self.state
            commanded_input = self.step_control(cmd, state, target_velocity, target_acceleration, cur_state, cur_velocity, cmd, dt, terrains=terrains)
            inputs_agg.append(commanded_input)
            rate.sleep()

        self.estimate_parameters_kernel(self.buffer)

        print(self.d_estimator.predict(np.array((0, 1))))

    def estimate_parameters_leastsq(self, buffer):
        if len(buffer) < 4:
            return

        for i in range(len(self.k_est), len(buffer)):
            buffer_state = buffer[i]
            xdot, ydot, theta_dot = (buffer_state[1] - buffer_state[0])[:3] / buffer_state[2]
            theta = (buffer_state[0] + buffer_state[1])[2] / 2
            v, w = buffer_state[3]
            self.d_est = np.append(self.d_est, [v*np.cos(theta), v*np.sin(theta)])
            self.d_goal = np.append(self.d_goal, [xdot, ydot])
            self.k_est = np.append(self.k_est, [w])
            self.k_goal = np.append(self.k_goal, [theta_dot])

        self.d = np.dot(self.d_est, self.d_goal) / np.linalg.norm(self.d_est)**2
        self.k = np.dot(self.k_est, self.k_goal) / np.linalg.norm(self.k_est)**2
        
        # print("residual d", np.linalg.norm(self.d * self.d_est - self.d_goal, axis=-1).mean())

    def estimate_parameters_gp(self, buffer):
        for i in range(len(self.d_est), len(buffer)):
            buffer_state = buffer[i]
            xdot, ydot, theta_dot = (buffer_state[1] - buffer_state[0])[:3] / buffer_state[2]
            theta = (buffer_state[0] + buffer_state[1])[2] / 2
            v, w = buffer_state[3]
            self.d_est = np.append(self.d_est, [v*np.cos(theta), v*np.sin(theta)])
            self.d_goal = np.append(self.d_goal, [xdot, ydot])
            self.k_est = np.append(self.k_est, [w])
            self.k_goal = np.append(self.k_goal, [theta_dot])

    def estimate_parameters_kernel(self, buffer):
        X = []
        y_d = []
        y_k = []
        for buffer_state in buffer:
            xdot, ydot, theta_dot = (buffer_state[1] - buffer_state[0])[:3] / buffer_state[2]
            theta = (buffer_state[0] + buffer_state[1])[2] / 2
            v, w = buffer_state[3]
            visual_features = buffer_state[4]
            
            d_val_x = xdot/(v*np.cos(theta))
            d_val_y = ydot/(v*np.sin(theta))
            k_val = theta_dot/w

            MAX_CAP = 2
            if not np.isnan(d_val_x) and not np.isnan(d_val_y) and not np.isnan(k_val) \
                and abs(d_val_x) < MAX_CAP and abs(d_val_y) < MAX_CAP and abs(k_val) < MAX_CAP:
                X.append(visual_features)
                y_d.append(np.mean((d_val_x, d_val_y)))
                y_k.append(k_val)

        X = np.array(X)
        y_d = np.array(y_d)
        y_k = np.array(y_k)

        if X.shape[0] > 0:
            self.d_estimator.reestimate(X, y_d)
            self.k_estimator.reestimate(X, y_k)

        self.buffer = []

    def step_control(self, cmd, target_position, target_velocity, target_acceleration, cur_position, cur_velocity, open_loop_input, dt, terrains=[]):
        """Specify a control law. For the grad/EC portion, you may want
        to edit this part to write your own closed loop controller.
        Note that this class constantly subscribes to the state of the robot,
        so the current configuratin of the robot is always stored in the 
        variable self.state. You can use this as your state measurement
        when writing your closed loop controller.

        Parameters
        ----------
            target_position : target position at the current step in
                              [x, y, theta, phi] configuration space.
            open_loop_input : the prescribed open loop input at the current
                              step, as a [u1, u2] pair.
        Returns:
            None. It simply sends the computed command to the robot.
        """
        theta = cur_position[2]
        v = self.vel_cmd if self.vel_cmd else open_loop_input[0]

        if abs(v) > 0.01:
            A_inv = np.array([[np.cos(theta)/self.d, np.sin(theta)/self.d],
                            [-np.sin(theta)/(v*self.k), np.cos(theta)/(v*self.k)]])
        else:
            A_inv = np.array([[np.cos(theta)/self.d, np.sin(theta)/self.d],
                              [0, 0]])

        # print(target_position[:2], target_position[:2] - cur_position[:2])
        # print(target_velocity[:2], cur_velocity[:2])
        taus = target_acceleration[:2] + 1.0 * (target_position[:2] - cur_position[:2]) + 0.5 * (target_velocity[:2] - cur_velocity[:2])
        control_input = np.matmul(A_inv, np.reshape(taus, (2,1)))
        self.vel_cmd += control_input[0] * dt
        control_input[0] = self.vel_cmd

        self.cmd(self.mock_velocity(cur_position, control_input, terrains))
        return control_input

    def cmd(self, msg):
        """
        Sends a command to the turtlebot / turtlesim

        Parameters
        ----------
        msg : numpy.ndarray
        """
        self.pub.publish(UnicycleCommandMsg(*msg))

    def subscribe(self, msg):
        """
        callback fn for state listener.  Don't call me...
        
        Parameters
        ----------
        msg : :obj:`UnicycleStateMsg`
        """
        self.state = np.array([msg.x, msg.y, msg.theta])

    def shutdown(self):
        rospy.loginfo("Shutting Down")
        self.cmd((0, 0))
