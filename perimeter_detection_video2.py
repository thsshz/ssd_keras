from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
import click
import os
import time
import cv2
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
import tensorflow as tf
from tensorflow.python.ops import gen_image_ops
tf.image.non_max_suppression = gen_image_ops.non_max_suppression_v2

def enhance_image(image):
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            sum_color = int(image[row][col][0]) + int(image[row][col][1]) + int(image[row][col][2])
            if sum_color >= 10:
                for channel in range(image.shape[2]):
                    new_color = int(image[row][col][channel]) + 50
                    if new_color > 255:
                        new_color = 255
                    image[row][col][channel] = new_color
    return image

def perimeter_detection(weights_path, video_path, result_path, threshold, perimeter_a, perimeter_b):
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
                    confidence_thresh=0.1,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)
    model.load_weights(weights_path, by_name=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    original_images = []
    process_images = []
    cap = cv2.VideoCapture(video_path)
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            transposed_frame = cv2.transpose(frame)
            transposed_frame = cv2.flip(transposed_frame, 1)
            original_images.append(transposed_frame)
            k = cv2.waitKey(20)
            if k & 0xff == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    for k in range(8250):
        sub_image = cv2.imread('perimeter_detection/sub_images/sub_' + str(k) + '.jpg')
        resize_image = cv2.resize(sub_image, (img_height, img_width))
        process_images.append(resize_image)
    print(len(original_images))
    process_images = np.array(process_images)
    # start_time = time.time()
    y_pred = model.predict(process_images, batch_size=8)
    # end_time = time.time()
    # print(end_time - start_time)
    confidence_threshold = 0.1

    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print('   class   conf xmin   ymin   xmax   ymax')

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    result_video = cv2.VideoWriter('result1.avi', fourcc, 25.0, (original_images[0].shape[0], original_images[0].shape[1]))
    
    vector_a = np.array([perimeter_a[0] - perimeter_b[0], perimeter_a[1] - perimeter_b[1]])
    distance_a = np.linalg.norm(vector_a)
    for k in range(len(y_pred_thresh)):
        print(k)
        print(y_pred_thresh[k])
        classes = ['background',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
        for box in y_pred_thresh[k]:
            if box[0] != 15:
                continue
            xmin = box[2] * 720 / 300
            ymin = box[3] * 400 / 300 + 600
            xmax = box[4] * 720 / 300
            ymax = box[5] * 400 / 300 + 600
            if xmin < 400:
                continue
            vector_b = np.array([xmin - perimeter_a[0], ymin - perimeter_a[1]])
            vector_cross = np.cross(vector_a, vector_b)
            distance = np.linalg.norm(vector_cross / distance_a)
            if vector_cross >= 0 or distance < threshold:
                cv2.rectangle(original_images[k], (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                continue
            vector_b = np.array([xmin - perimeter_a[0], ymax - perimeter_a[1]])
            vector_cross = np.cross(vector_a, vector_b)
            distance = np.linalg.norm(vector_cross / distance_a)
            if vector_cross >= 0 or distance < threshold:
                cv2.rectangle(original_images[k], (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                continue
            vector_b = np.array([xmax - perimeter_a[0], ymin - perimeter_a[1]])
            vector_cross = np.cross(vector_a, vector_b)
            distance = np.linalg.norm(vector_cross / distance_a)
            if vector_cross >= 0 or distance < threshold:
                cv2.rectangle(original_images[k], (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                continue
            vector_b = np.array([xmax - perimeter_a[0], ymax - perimeter_a[1]])
            vector_cross = np.cross(vector_a, vector_b)
            distance = np.linalg.norm(vector_cross / distance_a)
            if vector_cross >= 0 or distance < threshold:
                cv2.rectangle(original_images[k], (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                continue
            cv2.rectangle(original_images[k], (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.line(original_images[k], (int(perimeter_a[0]), int(perimeter_a[1])), (int(perimeter_b[0]), int(perimeter_b[1])), (0, 255, 255), 2)
        cv2.imwrite(result_path + '/detection_' + str(k) + '.jpg', original_images[k])
        result_image = original_images[k]
        transposed_image = cv2.transpose(result_image)
        transposed_image = cv2.flip(transposed_image, 0)
        result_video.write(transposed_image)
    result_video.release()
    cv2.destroyAllWindows()
    '''
    # start_time = time.time()
    vector_a = np.array([perimeter_a[0] - perimeter_b[0], perimeter_a[1] - perimeter_b[1]])
    distance_a = np.linalg.norm(vector_a)
    for k in range(len(y_pred_thresh)):
        print(file_names[k])
        print(y_pred_thresh[k])
        plt.figure(figsize=(12, 8))
        plt.imshow(original_images[k])
        plt.xticks([])
        plt.yticks([])
        current_axis = plt.gca()
        flag = 0
        for box in y_pred_thresh[k]:
            if box[0] != 15:
                continue
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = box[2] * original_images[k].shape[1] / img_width
            ymin = box[3] * original_images[k].shape[0] / img_height
            xmax = box[4] * original_images[k].shape[1] / img_width
            ymax = box[5] * original_images[k].shape[0] / img_height
            vector_b = np.array([xmin - perimeter_a[0], ymin - perimeter_a[1]])
            vector_cross = np.cross(vector_a, vector_b)
            distance = np.linalg.norm(vector_cross / distance_a)
            if vector_cross >= 0 or distance < threshold:
                flag = 1
                current_axis.add_patch(
                    plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='#FF0000', fill=False, linewidth=2))
                continue
            vector_b = np.array([xmin - perimeter_a[0], ymax - perimeter_a[1]])
            vector_cross = np.cross(vector_a, vector_b)
            distance = np.linalg.norm(vector_cross / distance_a)
            if vector_cross >= 0 or distance < threshold:
                flag = 1
                current_axis.add_patch(
                    plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='#FF0000', fill=False, linewidth=2))
                continue
            vector_b = np.array([xmax - perimeter_a[0], ymin - perimeter_a[1]])
            vector_cross = np.cross(vector_a, vector_b)
            distance = np.linalg.norm(vector_cross / distance_a)
            if vector_cross >= 0 or distance < threshold:
                flag = 1
                current_axis.add_patch(
                    plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='#FF0000', fill=False, linewidth=2))
                continue
            vector_b = np.array([xmax - perimeter_a[0], ymax - perimeter_a[1]])
            vector_cross = np.cross(vector_a, vector_b)
            distance = np.linalg.norm(vector_cross / distance_a)
            if vector_cross >= 0 or distance < threshold:
                flag = 1
                current_axis.add_patch(
                    plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='#FF0000', fill=False, linewidth=2))
                continue
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='#00FF00', fill=False, linewidth=2))
        line = Line2D([perimeter_a[0], perimeter_b[0]], [perimeter_a[1], perimeter_b[1]], color='#000000')
        current_axis.add_line(line)
        # plt.plot([perimeter_a[0], perimeter_b[0]], [perimeter_a[1], perimeter_b[1]], 'k')
        plt.savefig(result_path + '/perimeter_' + file_names[k], format='jpg')
        plt.close('all')
    # end_time = time.time()
    # print(end_time - start_time)
    '''

@click.command()
@click.option('--weights_path', required=True, help='model weights path')
@click.option('--video_path', required=True, help='Video path for detection')
@click.option('--result_path', required=True, help='Result image path')
@click.option('--threshold', required=True, help='Distance threshold')
@click.option('--perimeter_ax', required=True, help='Perimeter vertex a coordinate X')
@click.option('--perimeter_ay', required=True, help='Perimeter vertex a coordinate Y')
@click.option('--perimeter_bx', required=True, help='Perimeter vertex b coordinate X')
@click.option('--perimeter_by', required=True, help='Perimeter vertex b coordinate Y')
def perimeter_detection_command(weights_path, video_path, result_path, threshold, perimeter_ax, perimeter_ay,
                                perimeter_bx, perimeter_by):
    perimeter_a = []
    perimeter_a.append(float(perimeter_ax))
    perimeter_a.append(float(perimeter_ay))
    perimeter_b = []
    perimeter_b.append(float(perimeter_bx))
    perimeter_b.append(float(perimeter_by))
    return perimeter_detection(weights_path, video_path, result_path, float(threshold), perimeter_a, perimeter_b)


if __name__ == '__main__':
    perimeter_detection_command()
