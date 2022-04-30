import cv2
import numpy as np
from .VoxelGrid import VoxelGrid
from feature_extractor import extract_features

class FeatureGrid(VoxelGrid):
    def __init__(self):
        super().__init__(voxel_size=0.2, pixels_per_voxel=50, feature_size=7)

    def compute_features(self, image):
        return extract_features(image)

class ImageGrid(VoxelGrid):
    def __init__(self):
        VoxelGrid.__init__(self, voxel_size=0.01, pixels_per_voxel=1, feature_size=3)

    def compute_features(self, image):
        average = image.astype(float).sum(axis=(0, 1))
        return average[:3] / np.sum(average[3])


imageGrid = ImageGrid()
featureGrid = FeatureGrid()

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

# def dostuff_old(points, image, trans, rot):
#     cv2.imshow('press SPACE to save the image', image)
#     if cv2.waitKey(1) == ord(' '):
#         name = 'images/' + random.choice(WORDS) + '-' + random.choice(WORDS) + '.jpg'
#         cv2.imwrite(name, image)
