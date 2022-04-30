import cv2
import numpy as np


def extract_features(img):
    # edge = cv2.Canny(img, 200, 250, L2gradient=True)
    # cv2.imshow('image', img)
    # cv2.imshow('edge', edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(img.shape)
    img_reshaped = img.reshape((img.shape[0] * img.shape[1], 4))
    filter = img_reshaped[:, 3] > 0
    filtered_img = img_reshaped[filter, :3]
    filtered_img = np.expand_dims(filtered_img, axis=1)
    print(filtered_img.shape)

    features = []
    features.extend(average_hsvs(filtered_img))
    features.extend(average_rgbs(filtered_img))
    # features.extend(gabor_filters(img))
    features.append(rms_contrast(filtered_img))
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


def average_hsvs(img):
    # Convert to hsv
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Averages for each of h, s, and v
    averages = [np.mean(img[:, :, i]) for i in range(3)]
    return averages


def average_rgbs(img):
    averages = [np.mean(img[:, :, i]) for i in range(3)]
    return averages

# Gets root mean square contrast, which is essentially standard deviation of intensities
def rms_contrast(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_grey.std()


print(extract_features(cv2.imread(
    "images/spongebob-transparent.png", cv2.IMREAD_UNCHANGED)))
# print(extract_features(cv2.imread("images/grass.jpg")))
