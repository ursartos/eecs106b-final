from __future__ import print_function
from collections import deque

import rospy
import message_filters
import ros_numpy
import tf2_ros
import tf

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, LaserScan
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, TransformStamped

import numpy as np
import cv2

from .analyze import dostuff, dosensorstuff

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
    def __init__(self, image_sub_topic,
                       cam_info_topic,
                       sensor_sub_topic,
                       rgb_frame,
                       callback):

        self.messages = deque([], 5)
        self.pointcloud_frame = None
        self.caminfo = rospy.wait_for_message(cam_info_topic, CameraInfo)
        self.intrinsic_matrix = get_camera_matrix(self.caminfo)

        self.rgb_frame = rgb_frame
        # self.depth_frame = depth_frame
        self.callback = callback

        self.listener = tf.TransformListener()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        image_sub = rospy.Subscriber(image_sub_topic, Image, self.image_callback)
        scan_sub = rospy.Subscriber(sensor_sub_topic, LaserScan, self.sensor_callback)

    def image_callback(self, image):
        try:
            rgb_image = ros_numpy.numpify(image)
        except Exception as e:
            rospy.logerr(e)
            return
        time = image.header.stamp
        # self.messages.appendleft((time, rgb_image))
        self.publish_image((time, rgb_image))

    def sensor_callback(self, scan):
        # self.messages.appendleft(scan)
        self.publish_sensor(scan)

    # def publish_once_from_queue(self):
    #     if self.messages:
    #         msg = self.messages.pop()
    #         if type(msg) == tuple:
    #             self.publish_image(msg)
    #         else:
    #             self.publish_sensor(msg)

    def publish_image(self, msg):
        time, image = msg
        try:
            trans, rot = self.listener.lookupTransform('odom', 'base_link', time)
            rot = tf.transformations.quaternion_matrix(rot)[:3, :3]
            T_world_base = (rot, trans)

            trans, rot = self.listener.lookupTransform(self.rgb_frame, 'odom', time)
            rot = tf.transformations.quaternion_matrix(rot)[:3, :3]
            T_rgb_world = (rot, trans)
        except (tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException):
            return
        dostuff(image, self.intrinsic_matrix, T_world_base, T_rgb_world, self.callback)

    def publish_sensor(self, msg):
        try:
            time = msg.header.stamp
            pose = self.tf_buffer.lookup_transform('odom', 'base_link', time)
        except (tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException):
            return

        dosensorstuff(msg, pose)
