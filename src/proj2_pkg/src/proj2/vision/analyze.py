import cv2
import numpy as np
from .VoxelGrid import VoxelGrid
from feature_extractor import extract_features

class FeatureGrid(VoxelGrid):
    def __init__(self):
        super().__init__(voxel_size=0.2, pixels_per_voxel=50, feature_size=7)

    def compute_features(self, image):
        return extract_features(image)

class ObstacleGrid(VoxelGrid):
    def __init__(self):
        VoxelGrid.__init__(self, voxel_size=0.2, pixels_per_voxel=50, feature_size=1)

class ImageGrid(VoxelGrid):
    def __init__(self):
        VoxelGrid.__init__(self, voxel_size=0.01, pixels_per_voxel=1, feature_size=3)

    def compute_features(self, images):
        images = images.astype(float)
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
    cv2.imshow('grid', imageGrid.grid)
    cv2.waitKey(1)


def dosensorstuff(msg, pose):
    """Process laserscan data."""
    obstacleGrid.update_from_laserscan(msg, pose)
    cv2.imshow('occupancy', obstacleGrid.grid_to_probability())
    cv2.waitKey(1)
