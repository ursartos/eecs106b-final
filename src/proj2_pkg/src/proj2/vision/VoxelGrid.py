"""Stuff for homography and voxel computation."""

import cv2
import numpy as np

# A list of points corresponding to the corners of the floor in the robot image,
# specified in the base link frame's coordinate system
POINTS = np.array([[ 2.5,    1.562,  0.   ],
                   [ 2.5,   -1.592,  0.   ],
                   [ 0.54,  -0.395,  0.   ],
                   [ 0.54,   0.369,  0.   ]])

GRID_X_MIN = -5
GRID_Y_MIN = -5
GRID_X_MAX = 5
GRID_Y_MAX = 5


# Modified from 106a lab
def project_points_float(points, cam_matrix):
    """Project points from 3D camera space to the 2D image space."""
    homo_pixel_coords = np.dot(cam_matrix, points)
    pixel_coords = homo_pixel_coords[0:2, :] / homo_pixel_coords[2, :]
    return pixel_coords.astype(np.float32)


class VoxelGrid(object):
    """A VoxelGrid is a grid of voxels, where each voxel has a feature vector.

    Feature vectors are created by receiving images and transformations,
    then running feature extraction.
    """

    def __init__(self, voxel_size, pixels_per_voxel, feature_size):
        """Initialize the grid."""
        self.voxel_size = voxel_size
        self.pixels_per_voxel = pixels_per_voxel

        self.g_w = int(round((GRID_X_MAX - GRID_X_MIN)/self.voxel_size))
        self.g_h = int(round((GRID_Y_MAX - GRID_Y_MIN)/self.voxel_size))
        self.grid = np.zeros((self.g_h, self.g_w, feature_size))

    def compute_features(self, image):
        """Compute features for an image."""
        raise NotImplementedError()

    def update(self, image, cam_matrix, T_world_base, T_rgb_world):
        """Update the grid based on new camera data.

        image: RGB image from camera
        cam_matrix: Intrinsic camera matrix
        T_world_base: Transformation from base coordinates to world coordinates
        T_rgb_world: Transformation from world coordinates to rgb camera
        """
        rot_base_world, trans_base_world = T_world_base
        rot_world_rgb, trans_world_rgb = T_rgb_world

        # Project the bounds of the camera into the world frame
        # Then determine the range of voxels the camera image covers
        points_t = np.matmul(rot_base_world, POINTS.T) + np.array(trans_base_world).reshape((3,1))
        x_min = np.floor(np.min(points_t[0]) / self.voxel_size) * self.voxel_size
        x_max = np.ceil(np.max(points_t[0]) / self.voxel_size) * self.voxel_size
        y_min = np.floor(np.min(points_t[1]) / self.voxel_size) * self.voxel_size
        y_max = np.ceil(np.max(points_t[1]) / self.voxel_size) * self.voxel_size

        # Number of voxels that the image covers
        voxels_x = int(round((x_max - x_min) / self.voxel_size))
        voxels_y = int(round((y_max - y_min) / self.voxel_size))

        W = voxels_x * self.pixels_per_voxel
        H = voxels_y * self.pixels_per_voxel

        # Set up points for computing homography of world -> camera
        rectangle = np.array([
            [x_min, y_min, 0], [x_min, y_max, 0],
            [x_max, y_max, 0], [x_max, y_min, 0],
        ])
        dest = [[0,0], [0, H], [W, H], [W, 0]]

        # Project the corners of the voxel range into the camera image
        # The homography will be found between these corners and corners of the reprojected image
        p_rgb = np.matmul(rot_world_rgb, np.array(rectangle).T) + np.array(trans_world_rgb).reshape((3,1))
        rect_image = project_points_float(p_rgb, cam_matrix).T
        dest = np.array(dest).astype(np.float32)

        # Reproject the original image. Use RGBA matrix for computing masks
        matrix = cv2.getPerspectiveTransform(rect_image, dest)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        warped = cv2.warpPerspective(image, matrix, (W, H), cv2.INTER_CUBIC)

        for x in range(0, voxels_x):
            for y in range(0, voxels_y):
                # Compute features for voxels
                gx = x + int(round((x_min - GRID_X_MIN)/self.voxel_size))
                gy = y + int(round((y_min - GRID_Y_MIN)/self.voxel_size))

                PPV = self.pixels_per_voxel
                img = warped[y*PPV:(y+1)*PPV, x*PPV:(x+1)*PPV]
                n = np.sum(img[:, :, 3])

                # Only handle valid regions in the reprojected image
                if n > 0 and gx >= 0 and gy >= 0 and gx < self.g_w and gy < self.g_h:
                    self.grid[gy, gx] = self.compute_features(img)

        return warped
