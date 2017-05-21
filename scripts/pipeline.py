import glob
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage.measurements import label

from features import Features
from training import TrainingData


def find_boxes(image: np.ndarray,
               training_data: TrainingData,
               y_start: int = 404,
               y_stop: int = 660,
               x_start=640,
               window_start: int = 64,
               window_step: int = 32,
               predict=True):
    boxes = []
    image_width = image.shape[1]
    for window_size in range(window_start, y_stop - y_start + 1, window_step):
        for x in range(x_start, image_width - window_size + 1, window_step):
            y0 = y_start
            y1 = y_start + window_size
            x0 = x
            x1 = x + window_size

            if predict:
                window = image[y0:y1, x0:x1, :]

                if window_size > window_start:
                    window = cv2.resize(window, (window_start, window_start))

                features = Features(training_data.options).extract(window)
                scaled_features = training_data.scaler.transform(features.reshape(1, -1))
                prediction = training_data.svc.predict(scaled_features)

                if prediction == 1:
                    boxes.append(((x0, y0), (x1, y1)))
            else:
                boxes.append(((x0, y0), (x1, y1)))

    return boxes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def process_image(training_data: TrainingData, threshold: int = 1):
    def inner_function(image):
        boxes = find_boxes(image, training_data)
        heatmap = apply_threshold(add_heat(np.zeros_like(image[:, :, 0]).astype(np.float), boxes), threshold)
        heatmap = np.clip(heatmap, 0, 255)
        labels = label(heatmap)
        return draw_labeled_bboxes(np.copy(image), labels)

    return inner_function


def draw_boxes(image, boxes):
    image = np.copy(image)

    for p0, p1 in boxes:
        cv2.rectangle(image, p0, p1, (0, 0, 255), 3)

    return image


if __name__ == '__main__':
    with open('../data/training.p', 'rb') as f:
        training_data = pickle.load(f)

    # visualize test images
    test_images = glob.glob('../test_images/*.jpg')
    test_images_count = len(test_images)
    fig, axes = plt.subplots(nrows=test_images_count, ncols=3, figsize=(6 * 1280 // 720, 2 * test_images_count))
    for test_image, i in zip(test_images, range(0, test_images_count)):
        image = cv2.imread(test_image)
        boxes = find_boxes(image, training_data)
        heatmap = np.clip(apply_threshold(add_heat(np.zeros_like(image[:, :, 0]).astype(np.float), boxes), 1), 0, 255)

        axes[i][0].imshow(cv2.cvtColor(draw_boxes(image, boxes), cv2.COLOR_BGR2RGB))
        axes[i][1].imshow(heatmap, cmap='hot')
        axes[i][2].imshow(cv2.cvtColor(draw_labeled_bboxes(np.copy(image), label(heatmap)), cv2.COLOR_BGR2RGB))
    fig.tight_layout()
    plt.savefig('../output_images/pipeline.jpg')

    # process video
    input_clip = VideoFileClip('../project_video.mp4')
    output_clip = input_clip.fl_image(process_image(training_data, threshold=1))
    output_clip.write_videofile('../output.mp4', audio=False)
