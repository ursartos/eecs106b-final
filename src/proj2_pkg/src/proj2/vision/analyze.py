import cv2
import numpy as np
from .VoxelGrid import VoxelGrid
from feature_extractor import extract_features

class FeatureGrid(VoxelGrid):
    def __init__(self):
        VoxelGrid.__init__(self, voxel_size=0.2, pixels_per_voxel=50, feature_size=7)
        self.grid.fill(np.nan)

    def compute_features(self, voxels):
        return extract_features(voxels)

class ObstacleGrid(VoxelGrid):
    def __init__(self):
        VoxelGrid.__init__(self, voxel_size=0.2, pixels_per_voxel=50, feature_size=1)

class ImageGrid(VoxelGrid):
    def __init__(self):
        VoxelGrid.__init__(self, voxel_size=0.01, pixels_per_voxel=1, feature_size=3)

    def compute_features(self, images):
        images = images.reshape((-1, 4)).astype(float)
        return images[:, :3] / images[:, 3:4]


imageGrid = ImageGrid()
featureGrid = FeatureGrid()
obstacleGrid = ObstacleGrid()


def dostuff(image, cam_matrix, T_world_base, T_rgb_world):
    """Process camera data.

    image: RGB image from camera
    cam_matrix: Intrinsic camera matrix
    T_world_base: Transformation from base coordinates to world coordinates
    T_rgb_world: Transformation from world coordinates to rgb camera
    """
    imageGrid.update(image, cam_matrix, T_world_base, T_rgb_world)
    warped = featureGrid.update(image, cam_matrix, T_world_base, T_rgb_world)
    cv2.imshow('image', warped)

    # cv2.imshow('grid', cv2.resize(grid, None, fx=20, fy=20, interpolation=cv2.INTER_NEAREST))
    # cv2.imshow('grid', imageGrid.grid)

    grid = obstacleGrid.grid_to_probability()
    grid = cv2.resize(grid, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    grid = grid.reshape((1000, 1000, 1))

    cv2.imshow('grid', imageGrid.grid * (1 - grid))
    cv2.waitKey(1)

    return featureGrid.grid


def dosensorstuff(msg, pose):
    """Process laserscan data."""
    obstacleGrid.update_from_laserscan(msg, pose)
    # grid = obstacleGrid.grid_to_probability()
    # cv2.imshow('occupancy', cv2.resize(grid, None, fx=20, fy=20, interpolation=cv2.INTER_NEAREST))
    # cv2.waitKey(1)
