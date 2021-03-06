"""Stuff for homography and voxel computation."""

import cv2
import numpy as np
import rospy
import tf

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

# Convert between probabity and log-odds.
def ProbabilityToLogOdds(p):
    return np.log(p / (1.0 - p))

def LogOddsToProbability(l):
    return 1.0 / (1.0 + np.exp(-l))

class VoxelGrid(object):
    """A VoxelGrid is a grid of voxels, where each voxel has a feature vector.

    Feature vectors are created by receiving images and transformations,
    then running feature extraction.
    """

    def __init__(self, voxel_size, pixels_per_voxel, feature_size):
        """Initialize the grid."""
        self.voxel_size = voxel_size
        self.pixels_per_voxel = pixels_per_voxel
        self.feature_size = feature_size

        self.g_w = int(round((GRID_X_MAX - GRID_X_MIN)/self.voxel_size))
        self.g_h = int(round((GRID_Y_MAX - GRID_Y_MIN)/self.voxel_size))
        self.grid = np.zeros((self.g_h, self.g_w, self.feature_size))

        self.occupied_update = ProbabilityToLogOdds(0.7)
        self.occupied_threshold = ProbabilityToLogOdds(0.97)
        self.free_update = ProbabilityToLogOdds(0.3)
        self.free_threshold = ProbabilityToLogOdds(0.03)

        self.random_downsample = 0.1

    def compute_features(self, voxels):
        """Compute features for images."""
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
        x_min = max(np.floor(np.min(points_t[0]) / self.voxel_size) * self.voxel_size, GRID_X_MIN)
        x_max = min(np.ceil(np.max(points_t[0]) / self.voxel_size) * self.voxel_size, GRID_X_MAX)
        y_min = max(np.floor(np.min(points_t[1]) / self.voxel_size) * self.voxel_size, GRID_Y_MIN)
        y_max = min(np.ceil(np.max(points_t[1]) / self.voxel_size) * self.voxel_size, GRID_Y_MAX)

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

        PPV = self.pixels_per_voxel
        voxels = np.lib.stride_tricks.as_strided(warped,
            shape=(voxels_y, voxels_x, PPV, PPV, 4),
            strides=(W * PPV*4, PPV*4, W*4, 4, 1)
        )

        voxels = voxels.reshape(-1, PPV, PPV, 4) # Concatenate all voxels

        start_x = int(round((x_min - GRID_X_MIN)/self.voxel_size))
        start_y = int(round((y_min - GRID_Y_MIN)/self.voxel_size))

        positions = np.array([(y,x)
                     for y in range(start_y, start_y + voxels_y)
                     for x in range(start_x, start_x + voxels_x)
                    ])

        # Compute features for voxels
        features = self.compute_features(voxels)

        # Filter out nan filters
        good = ~np.any(np.isnan(features), axis=1)
        features = features[good]
        positions = positions[good]

        rows, cols = positions.T
        self.grid[rows, cols] = features

        # Only handle valid regions in the reprojected image
        # if n > 0 and :
        #     self.grid[gy, gx] = self.compute_features(img)

        return warped

    def update_from_laserscan(self, msg, pose):
        """Update grid from laserscan data."""
        sensor_x = pose.transform.translation.x
        sensor_y = pose.transform.translation.y
        if abs(pose.transform.translation.z) > 0.05:
            rospy.logwarn("Turtlebot is not on ground plane.")

        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
            [pose.transform.rotation.x, pose.transform.rotation.y,
             pose.transform.rotation.z, pose.transform.rotation.w])
        if abs(roll) > 0.1 or abs(pitch) > 0.1:
            rospy.logwarn("Turtlebot roll/pitch is too large.")

        # Loop over all ranges in the LaserScan.
        for idx, r in enumerate(msg.ranges):
            # Randomly throw out some rays to speed this up.
            if np.random.rand() > self.random_downsample or np.isnan(r):
                continue

            # Get angle of this ray in fixed frame.
            angle = msg.angle_min + idx * msg.angle_increment

            # Throw out this point if it is too close or too far away.
            if r > msg.range_max:
                rospy.logwarn("Range %f > %f was too large.", r, msg.range_max)
                continue
            if r < msg.range_min:
                rospy.logwarn("Range %f < %f was too small.", r, msg.range_min)
                continue

            # Walk along this ray from the scan point to the sensor.
            # Update log-odds at each voxel along the way.
            # Only update each voxel once.
            scan_x = sensor_x + r * np.cos(yaw + angle)
            scan_y = sensor_y + r * np.sin(yaw + angle)
            voxel_x = int((scan_x - GRID_X_MIN) / self.voxel_size)
            voxel_y = int((scan_y - GRID_Y_MIN) / self.voxel_size)
            if voxel_x >= 0 and voxel_y >= 0 and voxel_x < self.g_w and voxel_y < self.g_h:
                self.grid[voxel_y, voxel_x] += self.occupied_update
                if self.grid[voxel_y, voxel_x] > self.occupied_threshold:
                    self.grid[voxel_y, voxel_x] = self.occupied_threshold

            step = self.voxel_size #2 * np.sqrt(self.voxel_size)
            for i in np.arange(1, r, step):
                scan_x = sensor_x + (r-i) * np.cos(yaw + angle)
                scan_y = sensor_y + (r-i) * np.sin(yaw + angle)
                voxel_x = int((scan_x - GRID_X_MIN) / self.voxel_size)
                voxel_y = int((scan_y - GRID_Y_MIN) / self.voxel_size)
                if voxel_x >= 0 and voxel_y >= 0 and voxel_x < self.g_w and voxel_y < self.g_h:
                    self.grid[voxel_y, voxel_x] += self.free_update
                    if self.grid[voxel_y, voxel_x] < self.free_threshold:
                        self.grid[voxel_y, voxel_x] = self.free_threshold

    def grid_to_probability(self):
        """Convert grid from log-odds to probabilities."""
        return LogOddsToProbability(self.grid)
