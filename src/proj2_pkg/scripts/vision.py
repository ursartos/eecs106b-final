#!/usr/bin/env python
import rospy

from proj2.vision import PointcloudProcess

def main():
    CAM_INFO_TOPIC = '/camera/rgb/camera_info'
    RGB_IMAGE_TOPIC = '/camera/rgb/image_color'
    POINTS_TOPIC = '/camera/depth_registered/points'
    SENSOR_TOPIC = '/scan'

    RGB_FRAME = '/camera_rgb_optical_frame'
    # DEPTH_FRAME = '/camera_depth_optical_frame'

    rospy.init_node('kinect_listener')
    process = PointcloudProcess(POINTS_TOPIC, RGB_IMAGE_TOPIC, CAM_INFO_TOPIC,
                                SENSOR_TOPIC, RGB_FRAME)
    r = rospy.Rate(1000)

    while not rospy.is_shutdown():
        process.publish_once_from_queue()
        r.sleep()

if __name__ == '__main__':
    main()
