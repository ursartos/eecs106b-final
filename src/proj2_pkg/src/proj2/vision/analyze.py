import cv2
import numpy as np

# A list of points corresponding to the corners of the floor in the robot image,
# specified in the base link frame's coordinate system
POINTS = np.array([[ 2.5,    1.562,  0.   ]
                   [ 2.5,   -1.592,  0.   ]
                   [ 0.54,  -0.395,  0.   ]
                   [ 0.54,   0.369,  0.   ]])
VOXEL_SIZE = 0.5
PIXELS_PER_VOXEL = 50

GRID_X_MIN = -10
GRID_Y_MIN = -10
GRID_X_MAX = 10
GRID_Y_MAX = 10

g_w = int(round((GRID_X_MAX - GRID_X_MIN)/VOXEL_SIZE))
g_h = int(round((GRID_Y_MAX - GRID_Y_MIN)/VOXEL_SIZE))
grid = np.zeros((g_h, g_w, 3))

PPV = PIXELS_PER_VOXEL

def dostuff_old(points, image, trans, rot):
    cv2.imshow('press SPACE to save the image', image)
    if cv2.waitKey(1) == ord(' '):
        name = 'images/' + random.choice(WORDS) + '-' + random.choice(WORDS) + '.jpg'
        cv2.imwrite(name, image)

def dostuff(image, cam_matrix, T_world_base, T_rgb_world):
    rot_base_world, trans_base_world = T_world_base
    rot_world_rgb, trans_world_rgb = T_rgb_world

    points_t = rot_base_world @ points.T + trans_base_world.reshape((3,1))
    x_min = np.floor(np.min(points_t[0]) / VOXEL_SIZE) * VOXEL_SIZE
    x_max = np.ceil(np.max(points_t[0]) / VOXEL_SIZE) * VOXEL_SIZE
    y_min = np.floor(np.min(points_t[1]) / VOXEL_SIZE) * VOXEL_SIZE
    y_max = np.ceil(np.max(points_t[1]) / VOXEL_SIZE) * VOXEL_SIZE

    voxels_x = int(round((x_max - x_min) / VOXEL_SIZE))
    voxels_y = int(round((y_max - y_min) / VOXEL_SIZE))

    W = voxels_x * PIXELS_PER_VOXEL
    H = voxels_y * PIXELS_PER_VOXEL

    rectangle = np.array([
        [x_min, y_min, 0], [x_min, y_max, 0],
        [x_max, y_max, 0], [x_max, y_min, 0],
    ])
    dest = [[0,0], [0, H], [W, H], [W, 0]]
    p_rgb = rot_world_rgb @ np.array(rectangle).T + trans_world_rgb.reshape((3,1))

    rect_image = project_points_float(p_depth, cam_matrix).T
    dest = np.array(dest).astype(np.float32)

    mask_like_img = np.ones(images[2].shape[:2], dtype=np.uint8)

    matrix = cv2.getPerspectiveTransform(rect_image, dest)
    warped = cv2.warpPerspective(images[2], matrix, (W, H))
    mask = cv2.warpPerspective(mask_like_img, matrix, (W, H))
    warped_img = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    for x in range(0, voxels_x):
        for y in range(0, voxels_y):
            gx = x + int(round((x_min - GRID_X_MIN)/VOXEL_SIZE))
            gy = y + int(round((y_min - GRID_Y_MIN)/VOXEL_SIZE))

            img = warped_img[ y*PPV:(y+1)*PPV, x*PPV:(x+1)*PPV ]
            m = mask[ y*PPV:(y+1)*PPV, x*PPV:(x+1)*PPV ]
            n = np.sum(m)

            if n > 0:
                average = img.sum(axis=(0,1)) / n
                grid[gy, gx] = average

    cv2.imshow('grid', grid)
    cv2.waitKey(1)

# Modified from 106a lab
def project_points_float(points, cam_matrix):
    homo_pixel_coords = np.dot(cam_matrix, points)
    pixel_coords = homo_pixel_coords[0:2, :] / homo_pixel_coords[2, :]
    return pixel_coords.astype(np.float32)
