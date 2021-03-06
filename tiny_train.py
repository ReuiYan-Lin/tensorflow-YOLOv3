# coding: utf-8

from model.head.yolov3 import YOLOV3
from model.head.tiny_yolov3 import Tiny_YOLOV3
import config as cfg
from utils.tiny_data import Data
import tensorflow as tf
import numpy as np
import os
import time
import logging
import argparse
from eval.tiny_evaluator import Evaluator


class Yolo_train(Evaluator):
    def __init__(self):
        self.__learn_rate_init = cfg.LEARN_RATE_INIT
        self.__learn_rate_end = cfg.LEARN_RATE_END
        self.__max_periods = cfg.MAX_PERIODS
        self.__warmup_periods = cfg.WARMUP_PERIODS
        self.__weights_dir = cfg.WEIGHTS_DIR
        self.__weights_init = cfg.WEIGHTS_INIT
        self.__time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        self.__log_dir = os.path.join(cfg.LOG_DIR, 'train', self.__time)
        self.__moving_ave_decay = cfg.MOVING_AVE_DECAY
        self.__train_data = Data('train')
        self.__steps_per_period = len(self.__train_data)
        
        with tf.name_scope('input'):
            self.__input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.__label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.__label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.__mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.__lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.__training = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope('learning_rate'):
            self.__global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.__warmup_periods * self.__steps_per_period, dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant(self.__max_periods * self.__steps_per_period, dtype=tf.float64, name='train_steps')
            self.__learn_rate = tf.cond(
                pred=self.__global_step < warmup_steps,
                true_fn=lambda: self.__global_step / warmup_steps * self.__learn_rate_init,
                false_fn=lambda: self.__learn_rate_end + 0.5 * (self.__learn_rate_init - self.__learn_rate_end) *
                                 (1 + tf.cos((self.__global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.__global_step, 1.0)
        
        yolo = Tiny_YOLOV3(self.__training)
        conv_mbbox, conv_lbbox, \
        pred_mbbox, pred_lbbox = yolo.build_nework(self.__input_data)

        load_var = tf.global_variables('yolo-v3-tiny')
        restore_dict = self.__get_restore_dict(load_var)

        self.__loss = yolo.loss(conv_mbbox, conv_lbbox,
                                pred_mbbox, pred_lbbox,
                                self.__label_mbbox, self.__label_lbbox,
                                self.__mbboxes, self.__lbboxes)
            
        print("optimizer")
        with tf.name_scope('optimizer'):
            moving_ave = tf.train.ExponentialMovingAverage(self.__moving_ave_decay).apply(tf.trainable_variables())
            optimizer = tf.train.AdamOptimizer(self.__learn_rate).minimize(self.__loss, var_list=tf.trainable_variables())
            with tf.control_dependencies([optimizer, global_step_update]):
                with tf.control_dependencies([moving_ave]):
                    self.__train_op = tf.no_op()
        print("load_save")
        with tf.name_scope('load_save'):
            self.__load = tf.train.Saver(restore_dict)
            self.__save = tf.train.Saver(tf.global_variables(), max_to_keep=self.__max_periods)
        print("summary")
        with tf.name_scope('summary'):
            self.__loss_ave = tf.Variable(0, dtype=tf.float32, trainable=False)
            tf.summary.scalar('loss_ave', self.__loss_ave)
            tf.summary.scalar('learn_rate', self.__learn_rate)
            self.__summary_op = tf.summary.merge_all()
            self.__summary_writer = tf.summary.FileWriter(self.__log_dir)
            self.__summary_writer.add_graph(tf.get_default_graph())

        self.__sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.__sess.run(tf.global_variables_initializer())
        logging.info('Restoring weights from:\t %s' % self.__weights_init)
        self.__load.restore(self.__sess, self.__weights_init)

        super(Yolo_train, self).__init__(self.__sess, self.__input_data, self.__training,pred_mbbox, pred_lbbox)

    def __get_restore_dict(self, global_variables):
        org_weights_mess = []
        G = tf.Graph()
        with G.as_default():
            load = tf.train.import_meta_graph(self.__weights_init + '.meta')
            with tf.Session(graph=G) as sess_G:
                load.restore(sess_G, self.__weights_init)
                for var in tf.global_variables():
                    var_name = var.op.name
                    var_name_mess = str(var_name).split('/')
                    var_shape = var.shape
                    if var_name_mess[0] != 'yolo-v3-tiny':
                        continue
                    org_weights_mess.append([var_name, var_shape])

        cur_weights_mess = []
        for var in global_variables:
            var_name = var.op.name
            var_name_mess = str(var_name).split('/')
            var_shape = var.shape
            if var_name_mess[0] != 'yolo-v3-tiny':
                continue
            cur_weights_mess.append([var_name, var_shape])

        org_weights_num = len(org_weights_mess)
        cur_weights_num = len(cur_weights_mess)

        for var in org_weights_mess:
            logging.info('\t' + str(var).ljust(50)) 
        for var in cur_weights_mess:
            logging.info('\t' + str(var).ljust(50)) 

        #assert cur_weights_num == org_weights_num
        logging.info('Number of weights that will load:\t%d' % org_weights_num)
        logging.info('Number of weights that will load:\t%d' % cur_weights_num)

        cur_to_org_dict = {}
        for index in range(cur_weights_num):
            org_name, org_shape = org_weights_mess[index]
            cur_name, cur_shape = cur_weights_mess[index]
            if cur_shape != org_shape:
                logging.info(org_weights_mess[index])
                logging.info(cur_weights_mess[index])
                #raise RuntimeError
                continue
            cur_to_org_dict[cur_name] = org_name
            logging.info('\t cur_name :' + str(cur_name).ljust(70) +  str(cur_shape))
            logging.info('\t org_name :' + str(org_name).ljust(70) +  str(cur_shape))

        name_to_var_dict = {var.op.name: var for var in global_variables}
        restore_dict = {cur_to_org_dict[cur_name]: name_to_var_dict[cur_name] for cur_name in cur_to_org_dict}
        return restore_dict
        
    # test
    def Tiny_train(self):
        print("Train")
        print_loss_iter = self.__steps_per_period // 10
        total_train_loss = 0.0
        for period in range(self.__max_periods):
            for batch_image, batch_label_mbbox, batch_label_lbbox,\
                batch_mbboxes, batch_lbboxes \
                    in self.__train_data:
                _, loss_val, global_step_val = self.__sess.run(
                    [self.__train_op, self.__loss, self.__global_step],
                    feed_dict={
                        self.__input_data: batch_image,
                        self.__label_mbbox: batch_label_mbbox,
                        self.__label_lbbox: batch_label_lbbox,
                        self.__mbboxes: batch_mbboxes,
                        self.__lbboxes: batch_lbboxes,
                        self.__training: True
                    }
                )
                if np.isnan(loss_val):
                   raise ArithmeticError('The gradient is exploded')
                total_train_loss += loss_val
                print("\rStep : {} / {}".format(global_step_val%print_loss_iter,print_loss_iter),end='')
                if int(global_step_val) % print_loss_iter != 0:
                    continue

                train_loss = total_train_loss / print_loss_iter
                total_train_loss = 0.0

                self.__sess.run(tf.assign(self.__loss_ave, train_loss))
                summary_val = self.__sess.run(self.__summary_op)
                self.__summary_writer.add_summary(summary_val, global_step_val)
                print('Period:\t%d\tstep:\t%d\ttrain_loss:\t%.4f' % (period, global_step_val, train_loss))
                logging.info('Period:\t%d\tstep:\t%d\ttrain_loss:\t%.4f' % (period, global_step_val, train_loss))

            saved_model_name = os.path.join(self.__weights_dir, 'tiny_yolo.ckpt-%d-%d' % (period, global_step_val))
            self.__save.save(self.__sess, saved_model_name)
            logging.info('Saved model:\t%s' % saved_model_name)
            
            if period % 5 == 0:
                APs = self.APs_voc(2007, False, False)
                for cls in APs:
                    AP_mess = 'AP for %s = %.4f\n' % (cls, APs[cls])
                    logging.info(AP_mess.strip())
                mAP = np.mean([APs[cls] for cls in APs])
                mAP_mess = 'mAP = %.4f\n' % mAP
                logging.info(mAP_mess.strip())
                print(mAP_mess.strip())

                saved_model_name = os.path.join(self.__weights_dir, 'tiny_yolo.ckpt-%d-%.4f' % (period, float(mAP)))
                self.__save.save(self.__sess, saved_model_name)
                logging.info('Saved model:\t%s' % saved_model_name)
        self.__summary_writer.close()
        self.__sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train model')
    parser.add_argument('--gpu', help='select a gpu for test', default='0', type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    log_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    logging.basicConfig(filename='log/train/' + log_time + '.log', format='%(filename)s %(asctime)s\t%(message)s',
                        level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S', filemode='w')
    logging.info('Using GPU:' + args.gpu)

    Yolo_train().Tiny_train()



