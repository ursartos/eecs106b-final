# def find_homography(img):
import cv2
import numpy as np

# Add points onto image to find corners of known AR tag
img = cv2.imread("images/derogate-iffiest-ar-tag.jpg")
# img = cv2.circle(img, (213, 376), radius=2, color=(0, 0, 255), thickness=-1)
# img = cv2.circle(img, (252, 378), radius=2, color=(0, 0, 255), thickness=-1)
# img = cv2.circle(img, (201, 393), radius=2, color=(0, 0, 255), thickness=-1)
# img = cv2.circle(img, (242, 398), radius=2, color=(0, 0, 255), thickness=-1)

# # Visualize image
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

srcPoints = np.array([(213, 376), (252, 378), (201, 393), (242, 398)])
dstPoints = np.array(
    [(213, 376), (252, 378), (201, 416), (240, 420)]) + [0, 200]

# print(img)

H, _ = cv2.findHomography(srcPoints, dstPoints, 0)
# print(H)
im_dst = cv2.warpPerspective(img, H, img.shape[:2])

cv2.imshow('dst', im_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
