# import numpy as np
# from scipy.stats import skew, kurtosis
# from skimage.feature import local_binary_pattern
# from skimage.feature import graycomatrix, graycoprops
#
#
# def calculate_frequency_features(fft_data):
#     """Calculate advanced frequency domain features"""
#     magnitude = np.abs(fft_data)
#     phase = np.angle(fft_data)
#
#     # Statistical features
#     features = []
#     for data in [magnitude, phase]:
#         features.extend([
#             np.mean(data),
#             np.std(data),
#             np.var(data),
#             skew(data.flatten()),
#             kurtosis(data.flatten()),
#             np.percentile(data, [25, 50, 75])
#         ])
#
#     return np.array(features)
#
#
# def calculate_texture_features(image):
#     """Calculate texture features using LBP and GLCM"""
#     # LBP features
#     lbp = local_binary_pattern(image, 8, 1, method='uniform')
#     lbp_hist = np.histogram(lbp, bins=59, range=(0, 59))[0]
#
#     # GLCM features
#     glcm = graycomatrix(image, [1], [0, 45, 90, 135], symmetric=True, normed=True)
#     glcm_features = []
#     for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
#         glcm_features.extend(graycoprops(glcm, prop).flatten())
#
#     return np.concatenate([lbp_hist, glcm_features])

import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops


def calculate_frequency_features(fft_data):
    """Calculate advanced frequency domain features"""
    magnitude = np.abs(fft_data)
    phase = np.angle(fft_data)

    # Statistical features
    features = []
    for data in [magnitude, phase]:
        features.extend([
            np.mean(data),
            np.std(data),
            np.var(data),
            float(skew(data.flatten())),
            float(kurtosis(data.flatten())),
            float(np.percentile(data, 25)),
            float(np.percentile(data, 50)),
            float(np.percentile(data, 75))
        ])

    return np.array(features, dtype=np.float32)


def calculate_texture_features(image):
    """Calculate texture features using LBP and GLCM"""
    # LBP features
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    lbp_hist = np.histogram(lbp, bins=59, range=(0, 59))[0]

    # GLCM features
    glcm = graycomatrix(image, [1], [0, 45, 90, 135], symmetric=True, normed=True)
    glcm_features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        glcm_features.extend(graycoprops(glcm, prop).flatten())

    # Convert to float32 to ensure consistency
    return np.concatenate([lbp_hist, np.array(glcm_features)]).astype(np.float32)