import cv2
import numpy as np


def extract_features(voxels):
    # edge = cv2.Canny(img, 200, 250, L2gradient=True)
    # cv2.imshow('image', img)
    # cv2.imshow('edge', edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(img.shape)
    # img_reshaped = img.reshape((img.shape[0] * img.shape[1], 4))
    # filter = img_reshaped[:, 3] > 0
    # filtered_img = img_reshaped[filter, :3]
    # filtered_img = np.expand_dims(filtered_img, axis=1)
    # print(filtered_img.shape)

    # Reshape to get 1D array of filters
    voxels_reshaped = voxels.reshape((voxels.shape[0], voxels.shape[1] * voxels.shape[2], 4))
    # Get filter
    all_features = []
    for img_reshaped in voxels_reshaped:
        filter = img_reshaped[:, 3] > 0
        filtered_img = img_reshaped[filter, :3]
        if filtered_img.size == 0:
            all_features.append([np.nan] * 7)
            continue

        filtered_img = np.expand_dims(filtered_img, axis=1)
        features = []
        features.extend(average_hsvs(filtered_img))
        features.extend(average_hsvs(filtered_img))
        features.extend(rms_contrast(filtered_img))
        all_features.append(features)

    return np.array(all_features)
        

    filter = voxels_reshaped[:, :, 3] > 0
    # Apply filter and reshape back
    # filtered_voxels = np.reshape(voxels_reshaped[filter, :], (voxels.shape[0], -1, 4))
    # Extra dimension is necessary for format conversions
    # filtered_voxels = np.expand_dims(filtered_voxels, axis=2)

    # Get necessary features
    # features = rms_contrast(filtered_voxels)
    # features = np.hstack((features, average_hsvs(filtered_voxels)))
    # features = np.hstack((features, average_rgbs(filtered_voxels)))
    # features.extend(gabor_filters(img))

    return features


def gabor_filters(img):
    # Applies several Gabor filters with varied hyperparameters
    filtered_images = []
    num_filters = 16

    for i in range(num_filters):
        # Gets kernel with specified parameters
        kernel = cv2.getGaborKernel(
            (15, 15), 3, np.pi * i / num_filters, 10, 0.5, 0)
        # Normalization
        kernel /= kernel.sum() * 1.0
        # Apply filter
        filtered = cv2.filter2D(img, -1, kernel)

        filtered_images.append(filtered)

    return filtered_images


# def average_hsvs(voxels):
#     # Convert to hsv
#     imgs = []
#     for img in voxels:
#         imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
#     hsvs = np.array(imgs)

#     # Averages for each of h, s, and v
#     averages = [np.mean(hsvs[:, :, :, i], axis=1) for i in range(3)]
#     return np.array(averages)[:, :, 0].T

# def average_rgbs(voxels):
#     # Average for each r, g, b
#     averages = [np.mean(voxels[:, :, :, i], axis=1) for i in range(3)]
#     return np.array(averages)[:, :, 0].T

# # Gets root mean square contrast, which is essentially standard deviation of intensities
# def rms_contrast(voxels):
#     # Convert to grayscale
#     imgs = []
#     for img in voxels:
#         imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#     # Calculate standard deviations
#     grayed = np.array(imgs)
#     return np.std(grayed, axis=1)

def average_hsvs(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape((-1, 3))
    return np.mean(img, axis=0)


def average_rgbs(img):
    print(img.shape)
    return np.mean(img, axis=0)

def rms_contrast(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(img_grey, axis=0)


# images = np.stack((cv2.imread(
#     "images/spongebob-transparent.png", cv2.IMREAD_UNCHANGED), cv2.imread(
#     "images/spongebob-transparent.png", cv2.IMREAD_UNCHANGED)))
# print(images.shape)
# print(cv2.imread("images/spongebob-transparent.png", cv2.IMREAD_UNCHANGED).shape)
# print(extract_features(images))
