# coding:utf-8

import sys
import os
sys.path.append(os.path.abspath('../..'))
import config as cfg
import numpy as np
from model.layers import *
#from model.backbone.darknet53 import darknet53
from model.backbone.mobilenet_v1 import mobilenet_v1_base,mobilenet_v1_arg_scope
from utils import tools


class Tiny_mobileNet_YOLOV3(object):
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
            darknet_route1 = end_points['Conv2d_11_pointwise']
            conv = end_points['Conv2d_13_pointwise']

        with tf.variable_scope('yolo_v3'):       
            # ----------**********---------- Detection branch of large object ----------**********----------
            conv_lobj_branch = depthwise_conv_pointwise_conv(conv,1024,[3,3],stride=1,scope='conv15')
            conv_lbbox = convolutional(name='conv_lbbox', input_data=conv_lobj_branch,
                                                filters_shape=(1, 1, 1024, self.__gt_per_grid * (self.__num_classes + 5)),
                                                training=self.__training, downsample=False, activate=False, bn=False)
            pred_lbbox = decode(name='pred_lbbox', conv_output=conv_lbbox,num_classes=self.__num_classes, stride=self.__strides[1])
            # ----------**********---------- Detection branch of large object ----------**********----------
            
            # ----------**********---------- up sample and merge features map ----------**********----------

            conv = upsample(name='upsample0', input_data=conv_lobj_branch)

            darknet_route1 = depthwise_conv_pointwise_conv(darknet_route1,512,[3,3],stride=1,scope='conv17')
            conv = route(name='route0', previous_output=darknet_route1, current_output=conv)

            # ----------**********---------- Detection branch of middle object ----------**********----------
            conv_mobj_branch = depthwise_conv_pointwise_conv(conv,1024,[3,3],stride=1,scope='conv18')
            conv_mbbox = convolutional(name='conv_mbbox', input_data=conv_mobj_branch,
                                                filters_shape=(1, 1, 1024, self.__gt_per_grid * (self.__num_classes + 5)),
                                                training=self.__training, downsample=False, activate=False, bn=False)
            pred_mbbox = decode(name='pred_mbbox', conv_output=conv_mbbox,num_classes=self.__num_classes, stride=self.__strides[0])
            # ----------**********---------- Detection branch of middle object ----------**********----------
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
