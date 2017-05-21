import pickle

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import util
from features import Features, FeaturesOptions, ColorOptions, ColorSpace, HogOptions, SpatialOptions

OPTIONS = FeaturesOptions(
    color_options=ColorOptions(
        color_space=ColorSpace.HSV,
        bins_count=32
    ),
    hog_options=HogOptions(
        color_space=ColorSpace.YCrCb,
        color_channels={0, 1, 2},
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    ),
    spatial_options=SpatialOptions(
        color_space=ColorSpace.YCrCb,
        size=(32, 32)
    ))


class TrainingData:
    def __init__(self, options: FeaturesOptions, scaler: StandardScaler, svc: LinearSVC):
        self.options = options
        self.scaler = scaler
        self.svc = svc


def extract_features(filename, options: FeaturesOptions):
    return Features(options).extract(cv2.imread(filename, cv2.IMREAD_COLOR))


if __name__ == '__main__':
    print('Loading training data...')

    vehicles, non_vehicles = util.load_training_data()

    print('Extracting features for {:d} images...'.format(len(vehicles) + len(non_vehicles)))

    vehicles_features = [extract_features(vehicle, OPTIONS) for vehicle in vehicles]
    non_vehicles_features = [extract_features(non_vehicle, OPTIONS) for non_vehicle in non_vehicles]
    features = np.vstack((vehicles_features, non_vehicles_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(features)
    X = X_scaler.transform(features)
    y = np.hstack((np.ones(len(vehicles_features)), np.zeros(len(non_vehicles_features))))
    test_size = 0.2
    random_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print('Training model with test split of {:2.0f}/{:2.0f}...'.format((1 - test_size) * 100, test_size * 100))

    svc = LinearSVC(loss='hinge')
    svc.fit(X_train, y_train)

    print('Test accuracy is {:.4f}. Saving training data...'.format(svc.score(X_test, y_test)))

    data = TrainingData(options=OPTIONS, scaler=X_scaler, svc=svc)

    with open('../data/training.p', 'wb') as f:
        pickle.dump(data, f)

    print('Done!')
