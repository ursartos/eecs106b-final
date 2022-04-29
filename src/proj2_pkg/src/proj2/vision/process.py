from __future__ import print_function
from collections import deque

import rospy
import message_filters
import ros_numpy
import tf

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, TransformStamped

import numpy as np
import cv2

from .analyze import dostuff

def get_camera_matrix(camera_info_msg):
    return np.array(camera_info_msg.K).reshape((3,3))

def numpy_to_pc2_msg(points):
    return ros_numpy.msgify(PointCloud2, points, stamp=rospy.Time.now(),
        frame_id='camera_depth_optical_frame')

class PointcloudProcess:
    """
    Wraps the processing of a pointcloud from an input ros topic and publishing
    to another PointCloud2 topic.
    """
    def __init__(self, points_sub_topic,
                       image_sub_topic,
                       cam_info_topic,
                       rgb_frame,
                       depth_frame):

        self.num_steps = 0

        self.messages = deque([], 5)
        self.pointcloud_frame = None
        self.caminfo = rospy.wait_for_message(cam_info_topic, CameraInfo)
        self.intrinsic_matrix = get_camera_matrix(self.caminfo)

        self.rgb_frame = rgb_frame
        self.depth_frame = depth_frame

        self.listener = tf.TransformListener()

        image_sub = message_filters.Subscriber(image_sub_topic, Image, image_callback)

    def image_callback(self, image):
        try:
            rgb_image = ros_numpy.numpify(image)
        except Exception as e:
            rospy.logerr(e)
            return
        self.num_steps += 1
        self.messages.appendleft(rgb_image)

    def publish_once_from_queue(self):
        if self.messages:
            image = self.messages.pop()
            try:
                trans, rot = self.listener.lookupTransform('odom', 'base_link', rospy.Time(0))
                rot = tf.transformations.quaternion_matrix(rot)[:3, :3]
                T_world_base = (rot, trans)

                trans, rot = self.listener.lookupTransform(self.rgb_frame, 'odom', rospy.Time(0))
                rot = tf.transformations.quaternion_matrix(rot)[:3, :3]
                T_rgb_world = (rot, trans)
            except (tf.LookupException,
                    tf.ConnectivityException,
                    tf.ExtrapolationException):
                return
            dostuff(image. self.intrinsic_matrix, T_world_base, T_rgb_world)
