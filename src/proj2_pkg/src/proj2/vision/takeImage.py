#! /usr/bin/env python
# rospy for the subscriber
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Imager():
    def __init__(self):
        self.bridge = CvBridge()
        self.save = True

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)

        # Subscribers
        rospy.Subscriber('/camera/rgb/image_color', Image, self.takeImage)

    def takeImage(self, data):
        """
        Goal: knowing that you are already in the correct position, take an image and return it
        Baxter topic: /cameras/right_hand_camera/image
        """
        # Convert ROS Image message to CV2 and save
        cv2_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv2.imwrite('camera_image.png', cv2_img)


if __name__ == '__main__':
    rospy.init_node('Imager')
    node = Imager()
