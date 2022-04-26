import cv2
import numpy as np

import random

WORDS = [w for w in open('/usr/share/dict/words').read().splitlines() if "'" not in w]

MIN_Y = 0.3

def dostuff_old(points, image, trans, rot):
    cv2.imshow('press SPACE to save the image', image)
    if cv2.waitKey(1) == ord(' '):
        name = 'images/' + random.choice(WORDS) + '-' + random.choice(WORDS) + '.jpg'
        cv2.imwrite(name, image)

def dostuff(points, image, cam_matrix, trans, rot):

    IDX2D = lambda i, j, dj: dj * i + j


    # while True:
    segmented_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    xyz = np.array((points['x'], points['y'], points['z'])).reshape((3, -1))

    # MIN_Y = input('enter y value')
    # z_data = xyz[1][ ~np.isnan(xyz[1])]
    # print(np.min(z_data), np.max(z_data))

    xyz = xyz[ :, xyz[1] > MIN_Y ] # Filter by height
    pixel_coords = project_points(xyz, cam_matrix, trans, rot)

    image_h, image_w = segmented_image.shape[:2]

    in_frame = ((0 <= pixel_coords[0]) & (pixel_coords[0] < image_w)
                & (0 <= pixel_coords[1]) & (pixel_coords[1] < image_h))

    #points = points[in_frame] # Point cloud data that is in the image
    pixel_coords = pixel_coords[:, in_frame] # Pixels from point cloud
    j, i = pixel_coords
    linearized_pixel_coords = IDX2D(i, j, segmented_image.shape[1])
    linearized_segmentation = segmented_image.reshape(-1)
    linearized_segmentation[linearized_pixel_coords] = 1

    cv2.imshow('image', image)

    kernel = np.ones((3, 3), np.uint8)
    segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel, iterations=3)

    cv2.imshow('image mask', segmented_image*255)

    masked_img = cv2.bitwise_and(image, image, mask=segmented_image)
    cv2.imshow('masked_image', masked_img)

    # Now process masked_img
    cv2.waitKey(1)

    # if cv2.waitKey(1) == ord('q'):
    #     break

# From 106a lab
def project_points(points, cam_matrix, trans, rot):
    points = np.dot(rot, points) + np.array(trans).reshape((3, 1))
    homo_pixel_coords = np.dot(cam_matrix, points)
    pixel_coords = homo_pixel_coords[0:2, :] / homo_pixel_coords[2, :]
    pixel_coords = np.floor(pixel_coords).astype('int32')
    return pixel_coords
