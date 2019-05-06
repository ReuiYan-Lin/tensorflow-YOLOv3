# coding: utf-8

import numpy as np
import config as cfg
import cv2
import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import shutil
from utils import tools
from eval import voc_eval


class Evaluator(object):
    def __init__(self, sess, input_data, training, pred_sbbox, pred_mbbox, pred_lbbox):
        self._train_input_sizes = cfg.TRAIN_INPUT_SIZES
        self._test_input_size = cfg.TEST_INPUT_SIZE
        self._classes = cfg.CLASSES
        self._num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))
        self._score_threshold = cfg.SCORE_THRESHOLD
        self._iou_threshold = cfg.IOU_THRESHOLD
        self._dataset_path = cfg.DATASET_PATH
        self._project_path = cfg.PROJECT_PATH

        self.__sess = sess
        self.__input_data = input_data
        self.__training = training
        self.__pred_sbbox = pred_sbbox
        self.__pred_mbbox = pred_mbbox
        self.__pred_lbbox = pred_lbbox

    def __predict(self, image, test_input_size, valid_scale):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        yolo_input = tools.img_preprocess2(image, None, (test_input_size, test_input_size), False)
        yolo_input = yolo_input[np.newaxis, ...]
        pred_sbbox, pred_mbbox, pred_lbbox = self.__sess.run(
            [self.__pred_sbbox, self.__pred_mbbox, self.__pred_lbbox],
            feed_dict={
                self.__input_data: yolo_input,
                self.__training: False
            }
        )
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self._num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self._num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self._num_classes))], axis=0)
        bboxes = self.__convert_pred(pred_bbox, test_input_size, (org_h, org_w), valid_scale)
        return bboxes

    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):

        pred_bbox = np.array(pred_bbox)

        pred_coor = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]


        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio


        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)

        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0


        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))


        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self._score_threshold

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes

    def get_bbox(self, image, multi_test=False, flip_test=False):

        if multi_test:
            test_input_sizes = self._train_input_sizes[::3]
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale = (0, np.inf)
                bboxes_list.append(self.__predict(image, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(image[:, ::-1, :], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = image.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(image, self._test_input_size, (0, np.inf))
        bboxes = tools.nms(bboxes, self._score_threshold, self._iou_threshold, method='nms')
        return bboxes

    def __APs_calc(self, iou_thresh=0.5, use_07_metric=False):

        filename = os.path.join(self._project_path, 'eval', 'results', 'VOC2007', 'Main', 'comp3_det_test_{:s}.txt')
        cachedir = os.path.join(self._project_path, 'eval', 'cache')
        #annopath = os.path.join(self._dataset_path, '2007_test', 'Annotations', '{:s}.xml')
        annopath = self._dataset_path + '/2007_test'+'/Annotations/' + '{:s}.xml'


        imagesetfile = os.path.join(self._dataset_path, '2007_test', 'ImageSets', 'Main', 'test.txt')
        APs = {}
        for i, cls in enumerate(cfg.CLASSES):
            rec, prec, ap = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)
            APs[cls] = ap
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)
        return APs

    def APs_voc(self, year=2007, multi_test=False, flip_test=False):

        assert (year == 2007 or year == 2012)
        test_set_path = os.path.join(self._dataset_path, '%d_test' % year)
        img_inds_file = os.path.join(test_set_path, 'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file, 'r') as f:
            txt = f.readlines()
            image_inds = [line.strip() for line in txt]
        det_results_path = os.path.join(self._project_path, 'eval', 'results', 'VOC%d' % year, 'Main')
        if os.path.exists(det_results_path):
            shutil.rmtree(det_results_path)
        os.makedirs(det_results_path)

        for class_name in self._classes:
            with open(os.path.join(det_results_path, 'comp3_det_test_' + class_name + '.txt'), 'w') as f:
                f.close()
        
        for image_ind in image_inds:
            image_path = os.path.join(test_set_path, 'JPEGImages', image_ind + '.jpg')
            image_path
            image = cv2.imread(image_path)
            print("\rimage_path : {}".format(image_path),end='')
            bboxes_pr = self.get_bbox(image, multi_test, flip_test)
            for bbox in bboxes_pr:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = self._classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                bbox_mess = ' '.join([image_ind, score, xmin, ymin, xmax, ymax]) + '\n'
                with open(os.path.join(det_results_path, 'comp3_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(bbox_mess)
        if year == 2007:
            return self.__APs_calc()
        else:
            return None

