import cv2
import rospy
from takeImage import startImages, stopImages


def main():

    rospy.init_node('vision_test')

    # Capture image and use CV to calculate offsets from AR tag
    startImages()
    rospy.sleep(2.0)
    stopImages()
    image_path = 'camera_image.png'
    cv2.imshow(image_path)