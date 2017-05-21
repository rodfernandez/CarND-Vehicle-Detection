import glob
import os
import zipfile
from urllib import request

import numpy as np

VEHICLES_URL = "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip"
NON_VEHICLES_URL = "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip"
VEHICLES_DIR = '../training_data/vehicles'
NON_VEHICLES_DIR = '../training_data/non-vehicles'


def download_training_data():
    vehicles_zip = request.urlretrieve(VEHICLES_URL)[0]
    non_vehicles_zip = request.urlretrieve(NON_VEHICLES_URL)[0]

    zipfile.ZipFile(vehicles_zip).extractall(VEHICLES_DIR)
    zipfile.ZipFile(non_vehicles_zip).extractall(NON_VEHICLES_DIR)

    return VEHICLES_DIR, NON_VEHICLES_DIR


def collage(src: np.ndarray, rows: int, columns: int):
    count, height, width, channels = src.shape

    assert count == rows * columns
    assert height > 0
    assert width > 0
    assert channels == 1 or channels == 3

    dst = np.full((rows * height, columns * width, channels), 255, dtype=np.uint8)

    for i in range(rows):
        for j in range(columns):
            dst[i * height: (i + 1) * height, j * width: (j + 1) * width] = src[i * columns + j]

    return dst


def load_training_data():
    if os.path.exists('../training_data') is False:
        download_training_data()

    vehicles = glob.glob(VEHICLES_DIR + '/**/*.png', recursive=True)
    non_vehicles = glob.glob(NON_VEHICLES_DIR + '/**/*.png', recursive=True)

    return vehicles, non_vehicles
