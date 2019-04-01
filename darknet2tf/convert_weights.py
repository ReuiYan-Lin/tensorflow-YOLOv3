# -*- coding: utf-8 -*-

import tensorflow as tf

from yolo_v3 import darknet
from yolo_v3_tiny import yolo_v3_tiny
import logging
import time
from utils import load_coco_names, load_weights

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'weights_file', '', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/darknet53_416.ckpt', 'Chceckpoint file')

IS_Tiny = False
coco_class_num = 80
log_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
logging.basicConfig(filename='log/' + log_time + '.log', format='%(filename)s %(asctime)s\t%(message)s',
                    level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S', filemode='w')

logging.info(FLAGS.ckpt_file) 


if __name__ == '__main__':
    
    
    inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])

    with tf.name_scope('summary'):
        summary_writer = tf.summary.FileWriter('./log')
        summary_writer.add_graph(tf.get_default_graph())
        
    if IS_Tiny:
        '''
        with tf.variable_scope('detector'):
            detections = yolo_v3_tiny(inputs, coco_class_num,
                           data_format=FLAGS.data_format)
            load_ops = load_weights(tf.global_variables(
                    scope='detector'), FLAGS.weights_file)               
        '''
        with tf.variable_scope('yolo-v3-tiny'):
            detections = yolo_v3_tiny(inputs, coco_class_num,data_format=FLAGS.data_format)
            
        logging.info('Trainable variable:')
        for var in tf.global_variables(scope='yolo-v3-tiny'):
            logging.info('\t' + str(var.op.name).ljust(50) + str(var.shape)) 
            
        load_ops = load_weights(tf.global_variables(scope='yolo-v3-tiny'), FLAGS.weights_file)    
        saver = tf.train.Saver(tf.global_variables(scope='yolo-v3-tiny'))
        
                    

    else:
        darknet(inputs, data_format=FLAGS.data_format)
        load_ops = load_weights(tf.global_variables(scope='darknet'), FLAGS.weights_file)
        saver = tf.train.Saver(tf.global_variables(scope='darknet'))
        
        logging.info('Trainable variable:')
        for var in tf.global_variables(scope='darknet'):
            logging.info('\t' + str(var.op.name).ljust(50) + str(var.shape)) 
        
   
    with tf.Session() as sess:
        sess.run(load_ops)

        save_path = saver.save(sess, save_path=FLAGS.ckpt_file)
        print('Model saved in path: {}'.format(save_path))

