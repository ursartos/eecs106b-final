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

TERRAIN_DIM = 1

class UnicycleModelController(object):
    def __init__(self):
        """
        Executes a plan made by the planner
        """
        self.pub = rospy.Publisher('/unicycle/cmd_vel', UnicycleCommandMsg, queue_size=10)
        self.sub = rospy.Subscriber('/unicycle/state', UnicycleStateMsg, self.subscribe)
        self.state = UnicycleStateMsg()
        # self.target_positions = []
        # self.actual_positions = []
        self.d = 1
        self.k = 1
        self.buffer = []
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

    def execute_plan(self, plan, terrains=[]):
        """
        Executes a plan made by the planner

        Parameters
        ----------
        plan : :obj: Plan. See configuration_space.Plan
        """
        if len(plan) == 0:
            return
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
        
        while not rospy.is_shutdown():
            t = (rospy.Time.now() - start_t).to_sec()
            # dt = t - prev_t
            # prev_t = t
            dt = plan.dt
            if t < plan.times[-1]:
                state, cmd = plan.get(t)
                next_state, next_cmd = plan.get(t+dt)
                prev_state, prev_cmd = plan.get(t-dt)
            elif t < plan.times[-1] + 0:
                cmd = cmd*0
            else:
                break
            current_terrain_vector = self.current_pos_to_terrain(state[:2], terrains)[0]
            # state_vel = (self.state - cur_state)/dt 
            target_acceleration = ((next_state - state)/dt - (state - prev_state)/dt)/dt
            target_velocity = ((next_state - state)/dt)
            # print("states", prev_state[0], state[0], next_state[0])
            # print("target velocity", target_velocity[0])
            # print("open loop input", cmd[0])

            if np.linalg.norm(self.state - cur_state) > 0:
                cur_velocity = (self.state - cur_state)/(t-prev_state_change_time)
                self.buffer.append((cur_state, self.state, t-prev_state_change_time, np.mean(inputs_agg, axis=0), current_terrain_vector))
                prev_state_change_time = t
                self.estimate_parameters(self.buffer) #, self.terrain)

            cur_state = self.state
            commanded_input = self.step_control(state, target_velocity, target_acceleration, cur_state, cur_velocity, cmd, dt)
            inputs_agg.append(commanded_input)
            rate.sleep()

    def estimate_parameters(self, buffer):
        if len(buffer) < 4:
            return

        for i in range(len(self.k_est), len(buffer)):
            buffer_state = buffer[i]
            xdot, ydot, theta_dot = (buffer_state[1] - buffer_state[0])[:3] / buffer_state[2]
            theta = (buffer_state[0] + buffer_state[1])[2] / 2
            v, w = buffer_state[3]
            print("xdot", xdot, "v", v)
            # print("v and xdot", v, xdot)
            self.d_est = np.append(self.d_est, [v*np.cos(theta), v*np.sin(theta)])
            self.d_goal = np.append(self.d_goal, [xdot, ydot])
            self.k_est = np.append(self.k_est, [w])
            self.k_goal = np.append(self.k_goal, [theta_dot])

        # print(self.d_est, self.d_goal)

        # self.d_est, self.d_goal, self.k_est, self.k_goal = np.squeeze(self.d_est), np.squeeze(self.d_goal), np.squeeze(self.k_est), np.squeeze(self.k_goal)
        self.d = np.dot(self.d_est, self.d_goal) / np.linalg.norm(self.d_est)**2
        self.k = np.dot(self.k_est, self.k_goal) / np.linalg.norm(self.k_est)**2
        
        print(self.d)
        
        # print("residual d", np.linalg.norm(self.d * self.d_est - self.d_goal, axis=-1).mean())

        # print(self.d, self.k)
        # self.target_positions = np.array(self.target_positions)
        # self.actual_positions = np.array(self.actual_positions)
        # self.cmd([0, 0])

    def dist(self, orig_pt, dir, target):
        """
        https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
        """
        # p1 = orig_pt
        # p2 = orig_pt + np.array([-dir[1], dir[0]])
        # p3 = target
        # return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
        if np.linalg.norm(target - orig_pt) < 0.001:
            return 0

        return np.dot(dir, target - orig_pt)/np.sqrt(np.dot(target - orig_pt, target - orig_pt))

    def step_control(self, target_position, target_velocity, target_acceleration, cur_position, cur_velocity, open_loop_input, dt):
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
        # print(v, theta)
        # print(np.cos(theta))
        # print(-np.sin(theta)/v)

        if abs(v) > 0.01:
            A_inv = np.array([[np.cos(theta)/self.d, np.sin(theta)/self.d],
                            [-np.sin(theta)/(v*self.k), np.cos(theta)/(v*self.k)]])
        else:
            A_inv = np.array([[np.cos(theta)/self.d, np.sin(theta)/self.d],
                              [0, 0]])

        # Kp = 0.1 * np.eye(2)
        taus = target_acceleration[:2] + 0.5 * (target_position[:2] - cur_position[:2]) + 0.5 * (target_velocity[:2] - cur_velocity[:2])
        # print("Target acceleration", taus, target_acceleration[:2])

        control_input = np.matmul(A_inv, np.reshape(taus, (2,1)))
        # print("inputs", control_input)

        self.vel_cmd += control_input[0] * dt
        # print("vel cmd", self.vel_cmd)
        # print("open loop", open_loop_input)
        control_input[0] = self.vel_cmd
        self.cmd(control_input)
        # print(control_input)
        return control_input

        # self.target_positions.append(target_position)
        # self.actual_nppositions.append(self.state)
    
        # # get x coordinate of the point in the robot's coordinate frame
        # robot_unit_vec = np.array([np.cos(self.state[2]), np.sin(self.state[2])])
        # forward_dist = self.dist(self.state[:2], robot_unit_vec, target_position[:2])
        # right_dist = self.dist(self.state[:2], np.array([robot_unit_vec[1], -robot_unit_vec[0]]), target_position[:2])
        # # get distance from line of unit vec to target position
        # Kp_side = 1.0
        # Kp_dtheta = 0.5 #1.5 #-1.0
        # Kp_dphi = 2.0 #1 #-1.0
        # Kp_vel = 4.0

        # dtheta = (target_position[2] - self.state[2])
        # target_steering = (Kp_dtheta * dtheta) * np.sign(open_loop_input[0]) + (Kp_side * right_dist if open_loop_input[0] > 0 else -Kp_side * right_dist)
        # # target_phi_dot = Kp_dphi*(target_steering - self.state[3]) + 0.4 * (target_position[3] - self.state[3]) / (open_loop_input[0] if open_loop_input[0] > 0 else 1)

        # print("vals", dtheta, target_steering, forward_dist, open_loop_input[1])

        # # print(forward_dist, right_dist, target_position[2] - self.state[2])
        # # print("phi", self.state[3], "theta", self.state[2], "target theta", target_position[2])
        # # if abs(open_loop_input[0] - 0) < 0.01:
        # #     open_loop_adjusted = (open_loop_input[0] + np.sign(open_loop_input[0])*0.01)
        # # else:
        # #     open_loop_adjusted = 100000000000
        # inp = (open_loop_input + 1.0*np.array([forward_dist*Kp_vel, target_steering])) #Kp_side * right_dist * np.sign(open_loop_input[0]) + Kp_dtheta * (target_position[2] - self.state[2])]))

        # inp[0] = min(inp[0], max(inp[0], -2.0), 2.0)
        # inp[1] = min(inp[1], max(inp[1], -3.0), 3.0)

        # self.cmd(inp)


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
