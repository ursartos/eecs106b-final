#!/usr/bin/env python
import rospy

from proj2.vision import PointcloudProcess
from std_srvs.srv import Empty as EmptySrv

def main():
    CAM_INFO_TOPIC = '/camera/rgb/camera_info'
    RGB_IMAGE_TOPIC = '/camera/rgb/image_color'
    # POINTS_TOPIC = '/camera/depth_registered/points'
    SENSOR_TOPIC = '/scan'
    GRID_PUB_TOPIC = '/feature_grid'

    RGB_FRAME = '/camera_rgb_optical_frame'
    # DEPTH_FRAME = '/camera_depth_optical_frame'

    rospy.init_node('kinect_listener')

    print('Waiting for converter/reset service ...')
    rospy.wait_for_service('/converter/reset')
    print('found!')
    reset = rospy.ServiceProxy('/converter/reset', EmptySrv)
    reset()

    process = PointcloudProcess(RGB_IMAGE_TOPIC, CAM_INFO_TOPIC, SENSOR_TOPIC,
                                GRID_PUB_TOPIC, RGB_FRAME)
    r = rospy.Rate(1000)

    while not rospy.is_shutdown():
        # process.publish_once_from_queue()
        r.sleep()

if __name__ == '__main__':
    main()
