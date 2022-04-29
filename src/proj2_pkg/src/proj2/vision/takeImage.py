#! /usr/bin/env python

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


class Imager():
    def __init__(self):
        self.bridge = CvBridge()
        self.save = True

        # Subscribers
        rospy.Subscriber('/camera/rgb/image_rect_color', Image, self.takeImage)

        rospy.spin()

    def takeImage(self, data):
        """
        Goal: knowing that you are already in the correct position, take an image and return it
        Baxter topic: /cameras/right_hand_camera/image
        """
        # Convert ROS Image message to CV2 and save
        cv2_img = self.bridge.imgmsg_to_cv2(data)
        cv2.imwrite('camera_image.png', cv2_img)

        H = np.array([[-1.20440728e-01, -4.24266083e-01, 1.39737270e+02], 
                [ 9.45813958e-17, -1.08275610e+00, 3.21578560e+02],
                [ 9.53190461e-19, -4.18819015e-03,  1.00000000e+00]])
        homographied = cv2.warpPerspective(cv2_img, H, (200, 200))
        cv2.imwrite('homographied_image.png', homographied)

        rospy.loginfo("Saved image")
        rospy.sleep(5)


if __name__ == '__main__':
    rospy.init_node('Imager')
    node = Imager()
