# coding:utf-8

import sys
import os
sys.path.append(os.path.abspath('../..'))
import config as cfg
import numpy as np
from model.layers import *
#from model.backbone.darknet53 import darknet53
from utils import tools


class Tiny_YOLOV3(object):
    def __init__(self, training):
        self.__training = training
        self.__classes = cfg.CLASSES
        self.__num_classes = len(cfg.CLASSES)
        self.__strides = np.array(cfg.STRIDES)
        self.__gt_per_grid = cfg.GT_PER_GRID
        self.__iou_loss_thresh = cfg.IOU_LOSS_THRESH

    def build_nework(self, input_data, val_reuse=False):
        with tf.variable_scope('yolo-v3-tiny', reuse=val_reuse):
            conv = convolutional(name='conv0', input_data=input_data, filters_shape=(3, 3, 3, 16),
                                 training=self.__training)
            conv = pool("pool0",input_data=conv)
            conv = convolutional(name='conv1', input_data=conv, filters_shape=(3, 3, 16, 32),
                                 training=self.__training)
            conv = pool("pool1",input_data=conv)
            conv = convolutional(name='conv2', input_data=conv, filters_shape=(3, 3, 32, 64),
                                 training=self.__training)
            conv = pool("pool2",input_data=conv)
            conv = convolutional(name='conv3', input_data=conv, filters_shape=(3, 3, 64, 128),
                                 training=self.__training)
            conv = pool("pool3",input_data=conv)
            conv_route_1 = convolutional(name='conv4', input_data=conv, filters_shape=(3, 3, 128, 256),
                                 training=self.__training)
            conv = pool("pool4",input_data=conv_route_1)
            conv = convolutional(name='conv5', input_data=conv, filters_shape=(3, 3, 256, 512),
                                 training=self.__training)
            conv = pool("pool5",input_data=conv,stride=(1, 1, 1, 1))
            conv = convolutional(name='conv6', input_data=conv, filters_shape=(3, 3, 512, 1024),
                                 training=self.__training)
            conv_route_2 = convolutional(name='conv7', input_data=conv, filters_shape=(1, 1, 1024, 256),
                                 training=self.__training) #14*14*256
            
            # ----------**********---------- Detection branch of large object ----------**********----------
            conv_lbbox = convolutional(name='conv8', input_data=conv_route_2, filters_shape=(3, 3, 256, 512),
                                 training=self.__training)
            conv_lbbox = convolutional(name='conv9', input_data=conv_lbbox,
                                       filters_shape=(1, 1, 512, self.__gt_per_grid * (self.__num_classes + 5)),
                                       training=self.__training, downsample=False, activate=False, bn=False)
            pred_lbbox = decode(name='pred_lbbox', conv_output=conv_lbbox,
                                num_classes=self.__num_classes, stride=self.__strides[1])
            # ----------**********---------- Detection branch of large object ----------**********----------
            
            # ----------**********---------- up sample and merge features map ----------**********----------
            conv = convolutional(name='conv10', input_data=conv_route_2, filters_shape=(1, 1, 256, 128),
                                 training=self.__training)
            conv = upsample(name='upsample0', input_data=conv)
            conv = route(name='route0', previous_output=conv_route_1, current_output=conv)
            # ----------**********---------- up sample and merge features map ----------**********----------
            
            # ----------**********---------- Detection branch of middle object ----------**********---------
            conv_mbbox = convolutional(name='conv11', input_data=conv, filters_shape=(3, 3, 256+128, 256),
                                 training=self.__training)
            conv_mbbox = convolutional(name='conv12', input_data=conv_mbbox,
                                       filters_shape=(1, 1, 256, self.__gt_per_grid * (self.__num_classes + 5)),
                                       training=self.__training, downsample=False, activate=False, bn=False)
            pred_mbbox = decode(name='pred_mbbox', conv_output=conv_mbbox,
                                num_classes=self.__num_classes, stride=self.__strides[0])
            # ----------**********---------- Detection branch of middle object ----------**********----------
       
        return conv_mbbox, conv_lbbox, pred_mbbox, pred_lbbox

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
             conv_mbbox, conv_lbbox,
             pred_mbbox, pred_lbbox,
             label_mbbox, label_lbbox,
             mbboxes, lbboxes):

        loss_mbbox = self.__loss_per_scale('loss_mbbox', conv_mbbox, pred_mbbox, label_mbbox, mbboxes,
                                           self.__strides[0])
        loss_lbbox = self.__loss_per_scale('loss_lbbox', conv_lbbox, pred_lbbox, label_lbbox, lbboxes,
                                           self.__strides[1])
        with tf.name_scope('loss'):
            loss =  loss_mbbox + loss_lbbox
        return loss
