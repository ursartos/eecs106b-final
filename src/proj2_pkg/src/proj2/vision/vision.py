from __future__ import print_function
from collections import deque

import rospy
import message_filters
import ros_numpy
import tf
import tf2_ros
import tf2_geometry_msgs

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, TransformStamped

import numpy as np
import cv2

from cv_bridge import CvBridge

def get_camera_matrix(camera_info_msg):
    return np.array(camera_info_msg.K).reshape((3,3))

def isolate_object_of_interest(points, image, cam_matrix, trans, rot):
    segmented_image = isolate_tennis_ball(image)
    points = segment_pointcloud(points, segmented_image, cam_matrix, trans, rot)
    return points

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
                       cam_info_topic):

        self.num_steps = 0

        self.messages = deque([], 5)
        self.pointcloud_frame = None
        points_sub = message_filters.Subscriber(points_sub_topic, PointCloud2)
        image_sub = message_filters.Subscriber(image_sub_topic, Image)
        caminfo_sub = message_filters.Subscriber(cam_info_topic, CameraInfo)

        self._bridge = CvBridge()
        self.listener = tf.TransformListener()
        
        # self.points_pub = rospy.Publisher(points_pub_topic, PointCloud2, queue_size=10)
        # self.image_pub = rospy.Publisher('segmented_image', Image, queue_size=10)
        
        ts = message_filters.ApproximateTimeSynchronizer([points_sub, image_sub, caminfo_sub],
                                                          10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

    def callback(self, points_msg, image, info):
        try:
            intrinsic_matrix = get_camera_matrix(info)
            rgb_image = ros_numpy.numpify(image)
            points = ros_numpy.numpify(points_msg)
        except Exception as e:
            rospy.logerr(e)
            return
        self.num_steps += 1
        self.messages.appendleft((points, rgb_image, intrinsic_matrix))

    def publish_once_from_queue(self):
        if self.messages:
            points, image, info = self.messages.pop()
            try:
                trans, rot = self.listener.lookupTransform(
                                                       '/camera_rgb_optical_frame',
                                                       '/camera_depth_optical_frame',
                                                       rospy.Time(0))
                rot = tf.transformations.quaternion_matrix(rot)[:3, :3]
            except (tf.LookupException,
                    tf.ConnectivityException, 
                    tf.ExtrapolationException):
                return

            cv2.imshow('image', image)
            cv2.waitKey(1)
            points_msg = numpy_to_pc2_msg(points)
            # self.points_pub.publish(points_msg)
            # print("Published segmented pointcloud at timestamp:",
                   # points_msg.header.stamp.secs)

def main():
    CAM_INFO_TOPIC = '/camera/rgb/camera_info'
    RGB_IMAGE_TOPIC = '/camera/rgb/image_color'
    POINTS_TOPIC = '/camera/depth_registered/image'

    rospy.init_node('realsense_listener')
    process = PointcloudProcess(POINTS_TOPIC, RGB_IMAGE_TOPIC, CAM_INFO_TOPIC)
    r = rospy.Rate(1000)

    while not rospy.is_shutdown():
        process.publish_once_from_queue()
        r.sleep()

if __name__ == '__main__':
    main()
