# coding:utf-8

# yolo

Is_Tiny_YOLOv3 = False
Mode = "YOLOv3"
TRAIN_INPUT_SIZES = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
#TRAIN_INPUT_SIZES = [288,320, 352, 384, 416]
TEST_INPUT_SIZE = 320
if Is_Tiny_YOLOv3:
    STRIDES = [16, 32]
else:   
    STRIDES = [8, 16, 32]
IOU_LOSS_THRESH = 0.5

# train
BATCH_SIZE = 8
LEARN_RATE_INIT = 1e-4
LEARN_RATE_END = 1e-6
WARMUP_PERIODS = 2
MAX_PERIODS = 100

GT_PER_GRID = 3
MOVING_AVE_DECAY = 0.9995
MAX_BBOX_PER_SCALE = 150

# test
SCORE_THRESHOLD = 0.2    # The threshold of the probability of the classes
IOU_THRESHOLD = 0.45     # The threshold of the IOU when implement NMS

# name and path
DATASET_PATH = 'D:/github/VOCdevkit/'
PROJECT_PATH = 'D:/github/Stronger-yolo-master/'
ANNOT_DIR_PATH = 'data'
WEIGHTS_DIR = 'weights'
if Mode == "Tiny_net":
    WEIGHTS_INIT = 'darknet2tf/saved_model/tiny_yolo.ckpt-21-11375'
elif Mode == "MobileNet":
    #WEIGHTS_INIT = "darknet2tf/saved_model/mobilenet_v1_1.0_224.ckpt"
    WEIGHTS_INIT = "weights/weight/mobileNet_yolo.ckpt-45-0.5329"
elif Mode == "YOLOv3":
    WEIGHTS_INIT = 'darknet2tf/saved_model/darknet53.ckpt'
    #WEIGHTS_INIT = 'weights/yolo.ckpt-49-0.8331'
LOG_DIR = 'log'
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

