from enum import IntEnum

import cv2
import numpy as np
import skimage.feature


class ColorSpace(IntEnum):
    GRAY = cv2.COLOR_BGR2GRAY,
    HLS = cv2.COLOR_BGR2HLS,
    HSV = cv2.COLOR_BGR2HSV,
    LAB = cv2.COLOR_BGR2LAB,
    LUV = cv2.COLOR_BGR2LUV,
    RGB = cv2.COLOR_BGR2RGB,
    YCrCb = cv2.COLOR_BGR2YCrCb,
    YUV = cv2.COLOR_BGR2YUV


def convert_color_space(image: np.ndarray, color_space: ColorSpace):
    assert image.shape[2] == 3

    if color_space in ColorSpace:
        return cv2.cvtColor(image, color_space)
    else:
        return np.copy(image)


class SpatialOptions:
    def __init__(self, color_space: ColorSpace = ColorSpace.RGB, size: (int, int) = (32, 32)):
        self.color_space = color_space
        self.size = size


class SpatialFeatures:
    def __init__(self, options: SpatialOptions = SpatialOptions()):
        self.options = options

    def extract(self, image: np.ndarray):
        image = convert_color_space(image, self.options.color_space)
        return cv2.resize(image, self.options.size).ravel()


class ColorOptions:
    def __init__(self,
                 color_space: ColorSpace = ColorSpace.RGB,
                 bins_count: int = 12,
                 bins_range: (int, int) = (0, 256)):
        self.color_space = color_space
        self.bins_count = bins_count
        self.bins_range = bins_range


class ColorFeatures:
    def __init__(self, options: ColorOptions = ColorOptions()):
        self.options = options

    def extract(self, image: np.ndarray):
        image = convert_color_space(image, self.options.color_space)

        if len(image.shape) < 3:
            channels = 1
        else:
            channels = image.shape[2]

        features = []

        for channel in range(channels):
            features.append(np.histogram(image[:, :, channel], self.options.bins_count, self.options.bins_range)[0])

        return np.ravel(features)


class HogOptions:
    def __init__(self,
                 color_space: ColorSpace = ColorSpace.GRAY,
                 color_channels: set = {0},
                 orientations: int = 9,
                 pixels_per_cell: (int, int) = (8, 8),
                 cells_per_block: (int, int) = (2, 2),
                 feature_vector=True):
        self.color_space = color_space
        self.color_channels = color_channels
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.feature_vector = feature_vector


class HogFeatures:
    def __init__(self, options: HogOptions = HogOptions()):
        self.options = options

    def extract(self, image: np.ndarray):
        image = convert_color_space(image, self.options.color_space)

        if len(image.shape) < 3:
            channels = range(1)
        else:
            channels = self.options.color_channels

        features = []

        for color_channel in channels:
            features.append(skimage.feature.hog(image[:, :, color_channel],
                                                orientations=self.options.orientations,
                                                pixels_per_cell=self.options.pixels_per_cell,
                                                cells_per_block=self.options.cells_per_block,
                                                visualise=False,
                                                transform_sqrt=True,
                                                feature_vector=self.options.feature_vector))

        return np.ravel(features)


class FeaturesOptions:
    def __init__(self, color_options: ColorOptions = None, hog_options: HogOptions = None,
                 spatial_options: SpatialOptions = None):
        self.color = color_options
        self.hog = hog_options
        self.spatial = spatial_options


class Features:
    def __init__(self, options: FeaturesOptions):
        self.options = options

    def extract(self, image: np.ndarray):
        assert image is not None
        assert len(image.shape) >= 2

        features = ()

        if self.options.spatial is not None:
            features += (SpatialFeatures(self.options.spatial).extract(image),)

        if self.options.color is not None:
            features += (ColorFeatures(self.options.color).extract(image),)

        if self.options.hog is not None:
            features += (HogFeatures(self.options.hog).extract(image),)

        return np.concatenate(features)
