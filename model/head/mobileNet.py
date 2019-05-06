# coding:utf-8

import sys
import os
sys.path.append(os.path.abspath('../..'))
import config as cfg
import numpy as np
from model.layers import *
import tensorflow as tf 
#from model.backbone.darknet53 import darknet53
#from keras.applications.mobilenet import MobileNet
from model.backbone.mobilenet_v1 import mobilenet_v1_base,mobilenet_v1_arg_scope
import model.backbone.mobilenet_v2 as mobilenet_v2
import utils.tools as tools


class MobileNet_YOLOV3(object):
    def __init__(self, training):
        self.__training = training
        self.__classes = cfg.CLASSES
        self.__num_classes = len(cfg.CLASSES)
        self.__strides = np.array(cfg.STRIDES)
        self.__gt_per_grid = cfg.GT_PER_GRID
        self.__iou_loss_thresh = cfg.IOU_LOSS_THRESH

    def build_nework(self, input_data, val_reuse=False):

        with tf.contrib.slim.arg_scope(mobilenet_v1_arg_scope(is_training=False)):
            logits, end_points = mobilenet_v1_base(input_data)
            darknet_route1 = end_points['Conv2d_5_pointwise']
            darknet_route2 = end_points['Conv2d_11_pointwise']
            conv = end_points['Conv2d_13_pointwise']
        with tf.variable_scope('yolo_v3'):
            
            conv = convolutional(name='conv52', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                          training=self.__training)
            conv = convolutional(name='conv53', input_data=conv, filters_shape=(3, 3, 512, 1024),
                                          training=self.__training)
            conv = convolutional(name='conv54', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                          training=self.__training)
            conv = convolutional(name='conv55', input_data=conv, filters_shape=(3, 3, 512, 1024),
                                          training=self.__training)
            conv = convolutional(name='conv56', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                          training=self.__training)
            
            # ----------**********---------- Detection branch of large object ----------**********----------
            conv_lobj_branch = convolutional(name='conv_lobj_branch', input_data=conv,
                                                      filters_shape=(3, 3, 512, 1024), training=self.__training)
            conv_lbbox = convolutional(name='conv_lbbox', input_data=conv_lobj_branch,
                                                filters_shape=(1, 1, 1024, self.__gt_per_grid * (self.__num_classes + 5)),
                                                training=self.__training, downsample=False, activate=False, bn=False)
            pred_lbbox = decode(name='pred_lbbox', conv_output=conv_lbbox,num_classes=self.__num_classes, stride=self.__strides[2])
            # ----------**********---------- Detection branch of large object ----------**********----------
            
            # ----------**********---------- up sample and merge features map ----------**********----------
            
            conv = convolutional(name='conv57', input_data=conv, filters_shape=(1, 1, 512, 256),
                                          training=self.__training)
            conv = upsample(name='upsample0', input_data=conv)
            conv = route(name='route0', previous_output=darknet_route2, current_output=conv)
            # ----------**********---------- up sample and merge features map ----------**********----------
            
            
            conv = convolutional('conv58', input_data=conv, filters_shape=(1, 1, 512+256, 256),
                                          training=self.__training)
            conv = convolutional('conv59', input_data=conv, filters_shape=(3, 3, 256, 512),
                                          training=self.__training)
            conv = convolutional('conv60', input_data=conv, filters_shape=(1, 1, 512, 256),
                                          training=self.__training)
            conv = convolutional('conv61', input_data=conv, filters_shape=(3, 3, 256, 512),
                                          training=self.__training)
            conv = convolutional('conv62', input_data=conv, filters_shape=(1, 1, 512, 256),
                                          training=self.__training)
            # ----------**********---------- Detection branch of middle object ----------**********----------
            conv_mobj_branch = convolutional(name='conv_mobj_branch', input_data=conv,
                                                      filters_shape=(3, 3, 256, 512), training=self.__training)
            conv_mbbox = convolutional(name='conv_mbbox', input_data=conv_mobj_branch,
                                                filters_shape=(1, 1, 512, self.__gt_per_grid * (self.__num_classes + 5)),
                                                training=self.__training, downsample=False, activate=False, bn=False)
            pred_mbbox = decode(name='pred_mbbox', conv_output=conv_mbbox,num_classes=self.__num_classes, stride=self.__strides[1])
            # ----------**********---------- Detection branch of middle object ----------**********----------
            # ----------**********---------- Detection branch of middle object ----------**********----------
    
            # ----------**********---------- up sample and merge features map ----------**********----------
            
            conv = convolutional(name='conv63', input_data=conv, filters_shape=(1, 1, 256, 128),
                                          training=self.__training)
            conv = upsample(name='upsample1', input_data=conv)
            conv = route(name='route1', previous_output=darknet_route1, current_output=conv)
            # ----------**********---------- up sample and merge features map ----------**********----------
    
            
            conv = convolutional(name='conv64', input_data=conv, filters_shape=(1, 1, 256+128, 128),
                                          training=self.__training)
            conv = convolutional(name='conv65', input_data=conv, filters_shape=(3, 3, 128, 256),
                                          training=self.__training)
            conv = convolutional(name='conv66', input_data=conv, filters_shape=(1, 1, 256, 128),
                                          training=self.__training)
            conv = convolutional(name='conv67', input_data=conv, filters_shape=(3, 3, 128, 256),
                                          training=self.__training)
            conv = convolutional(name='conv68', input_data=conv, filters_shape=(1, 1, 256, 128),
                                          training=self.__training)
    
            # ----------**********---------- Detection branch of small object ----------**********----------
            conv_sobj_branch = convolutional(name='conv_sobj_branch', input_data=conv,
                                                      filters_shape=(3, 3, 128, 256), training=self.__training)
            conv_sbbox = convolutional(name='conv_sbbox', input_data=conv_sobj_branch,
                                                filters_shape=(1, 1, 256, self.__gt_per_grid * (self.__num_classes + 5)),
                                                training=self.__training, downsample=False, activate=False, bn=False)
            pred_sbbox = decode(name='pred_sbbox', conv_output=conv_sbbox,num_classes=self.__num_classes, stride=self.__strides[0])
            
            # ----------**********---------- Detection branch of small object ----------**********----------
        
        
        return conv_sbbox, conv_mbbox, conv_lbbox, pred_sbbox, pred_mbbox, pred_lbbox
    def __focal(self, target, actual, alpha=1, gamma=2):
        focal = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal

    def __loss_per_scale(self, name, conv, pred, label, bboxes, stride):

        with tf.name_scope(name):
            conv_shape = tf.shape(conv)
            batch_size = conv_shape[0]
            output_size = conv_shape[1]
            input_size = stride * output_size
            conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                     self.__gt_per_grid, 5 + self.__num_classes))
            conv_raw_conf = conv[:, :, :, :, 4:5]
            conv_raw_prob = conv[:, :, :, :, 5:]

            pred_coor = pred[:, :, :, :, 0:4]
            pred_conf = pred[:, :, :, :, 4:5]

            label_coor = label[:, :, :, :, 0:4]
            respond_bbox = label[:, :, :, :, 4:5]
            label_prob = label[:, :, :, :, 5:-1]
            label_mixw = label[:, :, :, :, -1:]

            # 计算GIOU损失
            GIOU = tools.GIOU(pred_coor, label_coor)
            GIOU = GIOU[..., np.newaxis]
            input_size = tf.cast(input_size, tf.float32)
            bbox_wh = label_coor[..., 2:] - label_coor[..., :2]
            bbox_loss_scale = 2.0 - 1.0 * bbox_wh[0] * bbox_wh[1] / (input_size ** 2)
            GIOU_loss = respond_bbox * bbox_loss_scale * (1.0 - GIOU)

            # (2)计算confidence损失
            iou = tools.iou_calc3(pred_coor[:, :, :, :, np.newaxis, :],
                                  bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, : ])
            max_iou = tf.reduce_max(iou, axis=-1)
            max_iou = max_iou[:, :, :, :, np.newaxis]
            respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.__iou_loss_thresh, tf.float32)

            conf_focal = self.__focal(respond_bbox, pred_conf)

            conf_loss = conf_focal * (
                    respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                    +
                    respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            )

            # (3)计算classes损失
            prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
            loss = tf.concat([GIOU_loss, conf_loss, prob_loss], axis=-1)
            loss = loss * label_mixw
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3, 4]))
            return loss

    def loss(self,
             conv_sbbox, conv_mbbox, conv_lbbox,
             pred_sbbox, pred_mbbox, pred_lbbox,
             label_sbbox, label_mbbox, label_lbbox,
             sbboxes, mbboxes, lbboxes):

        loss_sbbox = self.__loss_per_scale('loss_sbbox', conv_sbbox, pred_sbbox, label_sbbox, sbboxes,
                                           self.__strides[0])
        loss_mbbox = self.__loss_per_scale('loss_mbbox', conv_mbbox, pred_mbbox, label_mbbox, mbboxes,
                                           self.__strides[1])
        loss_lbbox = self.__loss_per_scale('loss_lbbox', conv_lbbox, pred_lbbox, label_lbbox, lbboxes,
                                           self.__strides[2])
        with tf.name_scope('loss'):
            loss = loss_sbbox + loss_mbbox + loss_lbbox
        return loss
