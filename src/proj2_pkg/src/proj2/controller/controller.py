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
        self.debug_input = -0.8
        rospy.on_shutdown(self.shutdown)

    # def current_pos_to_terrain(self, pos, terrains):
    #     terrain_vec = np.zeros(TERRAIN_DIM + 1)
    #     for i, terrain in enumerate(terrains):
    #         terrain_corners = terrain[0]
    #         if terrain_corners[0] <= pos[0] <= terrain_corners[1] and terrain_corners[2] <= pos[1] <= terrain_corners[3]:
    #             terrain_vec[i + 1] = 1
    #             return terrain_vec, terrain[1], terrain[2]
    #     terrain_vec[0] = 1
    #     return terrain_vec, 1, 1

    def mock_velocity(self, pos, commanded, terrain_vector, terrains):
        d = 1
        k = 1

        terrain_idx = np.argmax(terrain_vector)
        if terrain_idx == 0:
            return commanded

        terrain = terrains[terrain_idx - 1]
        d, k = terrain[1][0], terrain[2][0]

        dd = np.random.normal(0, terrain[1][1]) if (terrain[1][1]) else 0
        dk = np.random.normal(0, terrain[2][1]) if (terrain[2][1]) else 0

        d += dd
        k += dk

        mocked = [commanded[0] * d, commanded[1] * k]
        return mocked

    def execute_plan(self, plan, terrain_vectors, terrain_map, terrain_map_res=1, mock_terrains=[]):
        """
        Executes a plan made by the planner

        Parameters
        ----------
        plan : :obj: Plan. See configuration_space.Plan
        """
        terrains = mock_terrains
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
        sys_id_period = 1
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

            # current_terrain_vector = self.current_pos_to_terrain(self.state[:2], terrains)[0] # eventually this will be made into vision-based
            rounded_coordinates = (terrain_map_res * self.state[:2]).astype(int)
            current_terrain_vector = terrain_vectors[tuple(rounded_coordinates)]
            est_d, est_k = terrain_map[tuple(rounded_coordinates)]

            target_acceleration = ((next_state - state)/dt - (state - prev_state)/dt)/dt
            target_velocity = ((next_state - state)/dt)

            # random tests #
            # target_acceleration=[0,0,0] 
            # target_velocity=(plan.positions[-1] - plan.positions[0])/plan.dt/len(plan)
            if np.linalg.norm(self.state - cur_state) > 0:
                cur_velocity = (self.state - cur_state)/(t-prev_state_change_time)
                sys_id_count += 1

                if sys_id_count % sys_id_period == 0:
                    self.buffer.append((cur_state, self.state, t-prev_state_change_time, np.mean(inputs_agg, axis=0), current_terrain_vector))
                    # self.estimate_parameters_kernel(self.buffer)
                    inputs_agg = []

                prev_state_change_time = t

            cur_state = self.state
            commanded_input = self.step_control(cmd, state, target_velocity, target_acceleration,
                                                cur_state, cur_velocity, cmd, dt, terrains=terrains,
                                                est_d=est_d, est_k=est_k, terrain_vector=current_terrain_vector)
            inputs_agg.append(commanded_input)
            rate.sleep()

        self.cmd([0, 0])
        self.debug_input *= -1.0
        self.estimate_parameters_kernel(self.buffer)

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
        
    def estimate_parameters_gp(self, buffer):
        for i in range(len(self.d_est), len(buffer)):
            buffer_state = buffer[i]
            xdot, ydot, theta_dot = (buffer_state[1] - buffer_state[0])[:3] / buffer_state[2]
            theta = (buffer_state[0] + buffer_state[1])[2] / 2.0
            v, w = buffer_state[3]
            self.d_est = np.append(self.d_est, [v*np.cos(theta), v*np.sin(theta)])
            self.d_goal = np.append(self.d_goal, [xdot, ydot])
            self.k_est = np.append(self.k_est, [w])
            self.k_goal = np.append(self.k_goal, [theta_dot])

    def estimate_parameters_kernel(self, buffer):
        X_d = []
        X_k = []
        y_d = []
        y_k = []
        for buffer_state in buffer:
            xdot, ydot, theta_dot = (buffer_state[1] - buffer_state[0])[:3] / buffer_state[2]
            theta = (buffer_state[0] + buffer_state[1])[2] / 2.0
            v, w = buffer_state[3]
            visual_features = buffer_state[4]

            MAX_CAP = 2
            
            measured_v_vec = np.squeeze(np.array([xdot, ydot]))
            measured_v = np.sqrt(xdot**2 + ydot**2)
            expected_velocity_vec = np.squeeze(np.array([v*np.cos(theta), v*np.sin(theta)]))
            d = np.dot(measured_v_vec, expected_velocity_vec)/np.dot(expected_velocity_vec, expected_velocity_vec)
            # d = measured_v/abs(v)
            k = theta_dot/w

            if (abs(d) < MAX_CAP and abs(v) > 0.1):
                X_d.append(visual_features)
                y_d.append(d)
            if (abs(k) < MAX_CAP and abs(w) > 0.1):
                X_k.append(visual_features)
                y_k.append(k)

            # component-wise #
            # d_val_x = xdot/(v*np.cos(theta))
            # d_val_y = ydot/(v*np.sin(theta))
            # k_val = theta_dot/w

            # MAX_CAP = 2
            # if not np.isnan(d_val_x) and not np.isnan(d_val_y) \
            #     and abs(d_val_x) < MAX_CAP and abs(d_val_y) < MAX_CAP:
            #     X_d.append(visual_features)
            #     y_d.append(np.mean((d_val_x, d_val_y)))
            # if np.isnan(d_val_x) and not np.isnan(d_val_y) \
            #     and abs(d_val_y) < MAX_CAP:
            #     X_d.append(visual_features)
            #     y_d.append(d_val_y)
            # if np.isnan(d_val_y) and not np.isnan(d_val_x) \
            #     and abs(d_val_x) < MAX_CAP:
            #     X_d.append(visual_features)
            #     y_d.append(d_val_x)
            # if not np.isnan(k_val) and abs(k_val) < MAX_CAP:
            #     X_k.append(visual_features)
            #     y_k.append(k_val)

        X_d = np.array(X_d)
        X_k = np.array(X_k)
        y_d = np.array(y_d)
        y_k = np.array(y_k)

        if X_d.shape[0] > 0:
            self.d_estimator.reestimate(X_d, y_d)
        if X_k.shape[0] > 0:
            self.k_estimator.reestimate(X_k, y_k)

        self.buffer = []

    def step_control(self, cmd, target_position, target_velocity, target_acceleration,
                     cur_position, cur_velocity, open_loop_input, dt, terrains=[],
                     est_d=1, est_k=1, terrain_vector=[]):
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

        self.k = est_k
        self.d = est_d
        if abs(v) > 0.01:
            A_inv = np.array([[np.cos(theta)/self.d, np.sin(theta)/self.d],
                            [-np.sin(theta)/(v*self.k), np.cos(theta)/(v*self.k)]])
        else:
            A_inv = np.array([[np.cos(theta)/self.d, np.sin(theta)/self.d],
                              [0, 0]])

        taus = target_acceleration[:2] + 0.5 * (target_position[:2] - cur_position[:2]) + 1.5 * (target_velocity[:2] - cur_velocity[:2])
        control_input = np.matmul(A_inv, np.reshape(taus, (2,1)))
        self.vel_cmd += control_input[0] * dt
        self.vel_cmd = max(min(self.vel_cmd, 2), -2)
        control_input[0] = self.vel_cmd

        # todo: remove
        # control_input[0] = self.debug_input if cur_position[0] > 0.9 else 0
        # control_input[1] = 3.0
        # print(control_input, np.linalg.norm(cur_velocity))

        self.cmd(self.mock_velocity(cur_position, control_input, terrain_vector, terrains))
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