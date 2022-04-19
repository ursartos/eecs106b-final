import cv2
import numpy as np

def extract_features(img_path):
    img = cv2.imread(img_path)
    
    features = []
    features.extend(average_hsvs(img))
    features.extend(average_rgbs(img))
    features.extend(gabor_filters(img))
    return features

def gabor_filters(img):
    # Applies several Gabor filters with varied hyperparameters
    filtered_images = []
    num_filters = 16

    for i in range(num_filters):
        # Gets kernel with specified parameters
        kernel = cv2.getGaborKernel((15, 15), 3, np.pi * i / num_filters, 10, 0.5, 0)
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
    
print(extract_features("images/bluffed-acclaiming.jpg"))