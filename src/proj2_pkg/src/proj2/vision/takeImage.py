#! /usr/bin/env python
# rospy for the subscriber
import cv2
from numpy.lib.npyio import save
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

# Instantiate CvBridge
bridge = CvBridge()
save_image = True
subscriber = None


def takeImage(data):
    """
    Goal: knowing that you are already in the correct position, take an image and return it
    Baxter topic: /cameras/right_hand_camera/image
    """
    global save_image
    if (save_image):
        print("Received an image!")
        try:
            # Convert ROS Image message to CV2 and save
            cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imwrite('./src/camera_image.png', cv2_img)
            exit

        except CvBridgeError as e:
            print(e)

def startImages():
    global subscriber 
    # Image topic
    image_topic = '/camera/rgb/image_color'
    # Subscriber setup
    subscriber = rospy.Subscriber(image_topic, Image, takeImage)

def stop_saving():
    global save_image
    global subscriber

    save_image = False
    subscriber.unregister()


if __name__ == '__main__':
    startImages()
