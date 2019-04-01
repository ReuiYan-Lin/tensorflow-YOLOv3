# coding: utf-8

import numpy as np
import config as cfg
import cv2
import os
import tensorflow as tf
from model.head.yolov3 import YOLOV3
from model.head.mobileNet import MobileNet_YOLOV3
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import argparse
from utils import tools
from eval.evaluator import Evaluator
from timeit import default_timer as timer

class Yolo_test(Evaluator):
    def __init__(self, test_weight):
        log_dir = os.path.join(cfg.LOG_DIR, 'test')
        moving_ave_decay = cfg.MOVING_AVE_DECAY
        test_weight_path = os.path.join(cfg.WEIGHTS_DIR, test_weight)

        with tf.name_scope('input'):
            input_data = tf.placeholder(dtype=tf.float32,shape=[None,None,None,3], name='input_data')
            training = tf.placeholder(dtype=tf.bool, name='training')
        if cfg.Mode == "YOLOv3":
            _, _, _, pred_sbbox, pred_mbbox, pred_lbbox = YOLOV3(training).build_nework(input_data)
        elif cfg.Mode == "MobileNet":
            _, _, _, pred_sbbox, pred_mbbox, pred_lbbox = MobileNet_YOLOV3(training).build_nework(input_data)
        with tf.name_scope('summary'):
            tf.summary.FileWriter(log_dir).add_graph(tf.get_default_graph())
        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(moving_ave_decay)
        self.__sess = tf.Session()
        saver = tf.train.Saver(ema_obj.variables_to_restore())
        saver.restore(self.__sess, test_weight_path)
        super(Yolo_test, self).__init__(self.__sess, input_data, training, pred_sbbox, pred_mbbox, pred_lbbox)

    def detect_image(self, image):
        original_image = np.copy(image)
        bboxes = self.get_bbox(image)
        image = tools.draw_bbox(original_image, bboxes, self._classes)
        self.__sess.close()
        return image

    def test(self, year=2007, multi_test=False, flip_test=False):
        APs = self.APs_voc(year, multi_test, flip_test)
        APs_file = os.path.join(self._project_path, 'eval', 'APs.txt')
        with open(APs_file, 'w') as f:
            for cls in APs:
                AP_mess = 'AP for %s = %.4f\n' % (cls, APs[cls])
                print(AP_mess.strip())
                f.write(AP_mess)
            mAP = np.mean([APs[cls] for cls in APs])
            mAP_mess = 'mAP = %.4f\n' % mAP
            print(mAP_mess.strip())
            f.write(mAP_mess)
        self.__sess.close()
        
    def detect_video(self,video_path):
        import cv2
        vid = cv2.VideoCapture(video_path)
        output_path = './output/result.mp4'
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps       = vid.get(cv2.CAP_PROP_FPS)
        video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        #prev_time = timer()
        while True:
            return_value, frame = vid.read()
            #image = Image.fromarray(frame)
            
            original_image = np.copy(frame)
            prev_time = timer()
            bboxes = self.get_bbox(frame)
            curr_time = timer()
            image = tools.draw_bbox(original_image, bboxes, self._classes)
            
            #image = self.detect_image(frame)
            result = np.asarray(image)
            #curr_time = timer()
            exec_time = curr_time - prev_time
            #prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            print("exec_time : {}".format(exec_time))
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.__sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test model')
    parser.add_argument('--test_weight', help='name of test weights file', default='', type=str)
    parser.add_argument('--gpu', help='select a gpu for test', default='0', type=str)
    parser.add_argument('-mt', help='multi scale test', dest='mt', action='store_true', default=False)
    parser.add_argument('-ft', help='flip test', dest='ft', action='store_true', default=False)
    parser.add_argument('-t07', help='test voc 2007', dest='t07', action='store_true', default=False)
    parser.add_argument('-t12', help='test voc 2012', dest='t12', action='store_true', default=False)
    parser.add_argument('-video', help='video detect', default='D:/github/video_dataset/0016E5.MXF') #D:/github/video_dataset/0016E5.MXF
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    T = Yolo_test(args.test_weight)
    if  args.video:
        T.detect_video(args.video)
    elif args.t07:
        T.test(2007, args.mt, args.ft)
    elif args.t12:
        T.test(2012, args.mt, args.ft)
    else:
        test_set_path = os.path.join(cfg.DATASET_PATH, '%d_test' % 2007)
        img_inds_file = os.path.join(test_set_path, 'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file, 'r') as f:
            txt = f.readlines()
            image_inds = [line.strip() for line in txt]
        image_ind = np.random.choice(image_inds)
        image_path = os.path.join(test_set_path, 'JPEGImages', image_ind + '.jpg')
        image = cv2.imread(image_path)
        image = T.detect_image(image)
        cv2.imwrite('detect_result.jpg', image)