from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
import click
import os
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss


def perimeter_detection(weights_path, image_path, result_path, threshold, perimeter_a, perimeter_b):
    img_height = 300
    img_width = 300
    K.clear_session()  # Clear previous models from memory.

    model = ssd_300(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                    # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)
    model.load_weights(weights_path, by_name=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    original_images = []
    process_images = []
    file_names = []
    for root, dirs, files in os.walk(image_path):
        for file in files:
            file_names.append(file)
            img_path = image_path + '/' + file
            original_images.append(imread(img_path))
            resize_image = image.load_img(img_path, target_size=(img_height, img_width))
            resize_image = image.img_to_array(resize_image)
            process_images.append(resize_image)
    process_images = np.array(process_images)

    y_pred = model.predict(process_images)
    confidence_threshold = 0.5

    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print('   class   conf xmin   ymin   xmax   ymax')

    for k in range(y_pred_thresh):
        print(file_names[k])
        print(y_pred_thresh[k])
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        classes = ['background',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']

        plt.figure(figsize=(12, 8))
        plt.imshow(original_images[k])
        plt.xticks([])
        plt.yticks([])
        current_axis = plt.gca()

        for box in y_pred_thresh[k]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = box[2] * original_images[k].shape[1] / img_width
            ymin = box[3] * original_images[k].shape[0] / img_height
            xmax = box[4] * original_images[k].shape[1] / img_width
            ymax = box[5] * original_images[k].shape[0] / img_height
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})

        plt.savefig(result_path + '/detection_' + file_names[k], format='jpg')



@click.command()
@click.option('--weights_path', required=True, help='model weights path')
@click.option('--image_path', required=True, help='Image path for detection')
@click.option('--result_path', required=True, help='Result image path')
@click.option('--threshold', required=True, help='Distance threshold')
@click.option('--perimeter_a', required=True, help='Perimeter vertex a coordinates')
@click.option('--perimeter_b', required=True, help='Perimeter vertex b coordinates')
def perimeter_detection_command(weights_path, image_path, result_path, threshold, perimeter_a, perimeter_b):
    return perimeter_detection(weights_path, image_path, result_path, threshold, perimeter_a, perimeter_b)


if __name__ == '__main__':
    perimeter_detection_command()