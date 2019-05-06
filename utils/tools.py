# coding: utf-8

import numpy as np
import tensorflow as tf
import random
import colorsys
import cv2


def sigmoid(arr):

    arr = np.array(arr, dtype=np.float128)
    return 1.0 / (1.0 + np.exp(-1.0 * arr))


def softmax(arr):

    arr = np.array(arr, dtype=np.float128)
    arr_exp = np.exp(arr)
    return arr_exp / np.expand_dims(np.sum(arr_exp, axis=-1), axis=-1)




def iou_calc(boxes1, boxes2):

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])


    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])


    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def GIOU(boxes1, boxes2):

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])


    intersection_left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    intersection_right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    intersection = tf.maximum(intersection_right_down - intersection_left_up, 0.0)
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    GIOU = IOU - 1.0 * (enclose_area - union_area) / enclose_area

    return GIOU

def nms(bboxes, score_threshold, iou_threshold, sigma=0.3, method='nms'):

    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = iou_calc1(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            assert method in ['nms', 'soft-nms']
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    return best_bboxes


def img_preprocess1(image, bboxes, target_shape, correct_box=True):

    h_target, w_target = target_shape
    h_org, w_org, _ = image.shape

    image = np.copy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (w_target, h_target))
    image = image / 255.0

    if correct_box:
        h_ratio = 1.0 * h_target / h_org
        w_ratio = 1.0 * w_target / w_org
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w_ratio
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h_ratio
        return image, bboxes
    return image


def img_preprocess2(image, bboxes, target_shape, correct_box=True):

    h_target, w_target = target_shape
    h_org, w_org, _ = image.shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    resize_ratio = min(1.0 * w_target / w_org, 1.0 * h_target / h_org)
    resize_w = int(resize_ratio * w_org)
    resize_h = int(resize_ratio * h_org)
    image_resized = cv2.resize(image, (resize_w, resize_h))

    image_paded = np.full((h_target, w_target, 3), 128.0)
    dw = int((w_target - resize_w) // 2)
    dh = int((h_target - resize_h) // 2)
    image_paded[dh:resize_h+dh, dw:resize_w+dw,:] = image_resized
    image = image_paded / 255.0

    if correct_box:
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
        return image, bboxes
    return image


def draw_bbox(original_image, bboxes, classes):

    num_classes = len(classes)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    image_h, image_w, _ = original_image.shape
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(1.0 * (image_h + image_w) // 600)
        cv2.rectangle(original_image, (coor[0], coor[1]), (coor[2], coor[3]), bbox_color, bbox_thick)

        bbox_mess = '%s: %.3f' % (classes[class_ind], score)
        text_loc = (int(coor[0]), int(coor[1] + 5) if coor[1] < 20 else int(coor[1] - 5))
        cv2.putText(original_image, bbox_mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image_h, (255, 255, 255), bbox_thick // 3)
    return original_image