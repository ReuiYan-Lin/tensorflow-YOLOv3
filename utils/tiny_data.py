# coding: utf-8

import sys
import os
sys.path.append(os.path.abspath('..'))
import cv2
import random
import tensorflow as tf
import logging
import numpy as np
import config as cfg
import utils.tools as tools
import utils.dataAug as dataAug
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class Data(object):
    def __init__(self, dataset_type, split_ratio=1.0):

        self.__annot_dir_path = cfg.ANNOT_DIR_PATH
        self.__train_input_sizes = cfg.TRAIN_INPUT_SIZES
        self.__strides = np.array(cfg.STRIDES)
        self.__batch_size = cfg.BATCH_SIZE
        self.__classes = cfg.CLASSES
        self.__num_classes = len(self.__classes)
        self.__gt_per_grid = cfg.GT_PER_GRID
        self.__class_to_ind = dict(zip(self.__classes, range(self.__num_classes)))
        self.__max_bbox_per_scale = cfg.MAX_BBOX_PER_SCALE

        annotations = self.__load_annotations(dataset_type)
        num_annotations = len(annotations)
        self.__annotations = annotations[: int(split_ratio * num_annotations)]
        self.__num_samples = len(self.__annotations)
        logging.info(('The number of image for %s is:' % dataset_type).ljust(50) + str(self.__num_samples))
        self.__num_batchs = np.ceil(self.__num_samples // self.__batch_size)
        self.__batch_count = 0

    def batch_size_change(self, batch_size_new):
        self.__batch_size = batch_size_new
        self.__num_batchs = np.ceil(self.__num_samples // self.__batch_size)
        logging.info('Use the new batch size: %d' % self.__batch_size)
    def __load_annotations(self, dataset_type):

        if dataset_type not in ['train', 'test']:
            raise ImportError("You must choice one of the 'train' or 'test' for dataset_type parameter")
        annotation_path = os.path.join(self.__annot_dir_path, dataset_type + '_annotation.txt')
        with open(annotation_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):
            self.__train_input_size = random.choice(self.__train_input_sizes)
            self.__train_output_sizes = self.__train_input_size // self.__strides

            batch_image = np.zeros((self.__batch_size, self.__train_input_size, self.__train_input_size, 3))
            batch_label_mbbox = np.zeros((self.__batch_size, self.__train_output_sizes[0], self.__train_output_sizes[0],
                                          self.__gt_per_grid, 6 + self.__num_classes))
            batch_label_lbbox = np.zeros((self.__batch_size, self.__train_output_sizes[1], self.__train_output_sizes[1],
                                          self.__gt_per_grid, 6 + self.__num_classes))
            batch_mbboxes = np.zeros((self.__batch_size, self.__max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.__batch_size, self.__max_bbox_per_scale, 4))
            num = 0
            if self.__batch_count < self.__num_batchs:
                while num < self.__batch_size:
                    index = self.__batch_count * self.__batch_size + num
                    if index >= self.__num_samples:
                        index -= self.__num_samples
                    annotation = self.__annotations[index]
                    image_org, bboxes_org = self.__parse_annotation(annotation)

                    # mixup
                    if random.random() < 0.5:
                        index_mix = random.randint(0, self.__num_samples - 1)
                        annotation_mix = self.__annotations[index_mix]
                        image_mix, bboxes_mix = self.__parse_annotation(annotation_mix)

                        lam = np.random.beta(1.5, 1.5)
                        image = lam * image_org + (1 - lam) * image_mix
                        bboxes_org = np.concatenate(
                            [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=-1)
                        bboxes_mix = np.concatenate(
                            [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=-1)
                        bboxes = np.concatenate([bboxes_org, bboxes_mix])
                    else:
                        image = image_org
                        bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=-1)
                    
                    label_mbbox, label_lbbox, mbboxes, lbboxes = self.__create_tiny_label(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
        
                self.__batch_count += 1
                return batch_image, batch_label_mbbox, batch_label_lbbox, \
                        batch_mbboxes, batch_lbboxes
            else:
                self.__batch_count = 0
                np.random.shuffle(self.__annotations)
                raise StopIteration

    def __parse_annotation(self, annotation):

        line = annotation.split()
        image_path = line[0]
        image = np.array(cv2.imread(image_path))
        bboxes = np.array([box.split(',') for box in line[1:]], dtype=int)

        image, bboxes = dataAug.random_horizontal_flip(np.copy(image), np.copy(bboxes))
        image, bboxes = dataAug.random_crop(np.copy(image), np.copy(bboxes))
        image, bboxes = dataAug.random_translate(np.copy(image), np.copy(bboxes))
        image, bboxes = tools.img_preprocess2(np.copy(image), np.copy(bboxes),
                                              (self.__train_input_size, self.__train_input_size), True)
        return image, bboxes
    
    def __create_tiny_label(self,bboxes):

        label = [np.zeros((self.__train_output_sizes[i], self.__train_output_sizes[i], self.__gt_per_grid,
                           6 + self.__num_classes)) for i in range(2)]
        # mixup weight位默认为1.0
        for i in range(2):
            label[i][:, :, :, -1] = 1.0
        bboxes_count = [np.zeros((self.__train_output_sizes[i], self.__train_output_sizes[i]))
                        for i in range(2)]

        bboxes_coor = [np.zeros((self.__max_bbox_per_scale, 4)) for _ in range(2)]
        bbox_count = np.zeros((2,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mixw = bbox[5]
            
            # label smooth
            onehot = np.zeros(self.__num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.__num_classes, 1.0 / self.__num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_scale = np.sqrt(np.multiply.reduce(bbox_coor[2:] - bbox_coor[:2]))
            # (xmin, ymin, xmax, ymax) -> (x_center, y_center)
            bbox_center = (bbox_coor[2:] + bbox_coor[:2]) * 0.5

            if bbox_scale <= 60:
                best_detect = 0
            else:
                best_detect = 1
                
            xind, yind = np.floor(1.0 * bbox_center / self.__strides[best_detect]).astype(np.int32)
            gt_count = int(bboxes_count[best_detect][yind, xind] % self.__gt_per_grid)
            if bboxes_count[best_detect][yind, xind] == 0:
                gt_count = [True for _ in range(self.__gt_per_grid)]
            label[best_detect][yind, xind, gt_count, :] = 0
            label[best_detect][yind, xind, gt_count, 0:4] = bbox_coor
            label[best_detect][yind, xind, gt_count, 4:5] = 1.0
            label[best_detect][yind, xind, gt_count, 5:-1] = smooth_onehot
            label[best_detect][yind, xind, gt_count, -1] = bbox_mixw
            bboxes_count[best_detect][yind, xind] += 1

            bbox_ind = int(bbox_count[best_detect] % self.__max_bbox_per_scale)
            bboxes_coor[best_detect][bbox_ind, :4] = bbox_coor
            bbox_count[best_detect] += 1
        label_mbbox, label_lbbox = label
        mbboxes, lbboxes = bboxes_coor
        return label_mbbox, label_lbbox, mbboxes, lbboxes
    

    def __len__(self):
        return int(self.__num_batchs)


if __name__ == '__main__':
    data_obj = Data('train')
    for batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes in data_obj:
        print(batch_image.shape)
        print(batch_label_sbbox.shape)
        print(batch_label_mbbox.shape)
        print(batch_label_lbbox.shape)
        print(batch_sbboxes.shape)
        print(batch_mbboxes.shape)
        print(batch_lbboxes.shape)

    data_obj = Data('test')
    for batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes in data_obj:
        print(batch_image.shape)
        print(batch_label_sbbox.shape)
        print(batch_label_mbbox.shape)
        print(batch_label_lbbox.shape)
        print(batch_sbboxes.shape)
        print(batch_mbboxes.shape)
        print(batch_lbboxes.shape)