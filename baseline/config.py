#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/12 17:04
# @Author  : Mr.D
# @Site    : 
# @File    : config.py
# @Software: PyCharm

# -----------------------------------------超参设置--------------------------
from keras.optimizers import SGD
import time


IMG_WIGTH = 64

IMG_HIGHT = 128

LEARNING_RATE = 0.001

OPTIMIZER = SGD(lr=LEARNING_RATE, momentum=0.9, decay=0.0, nesterov=True)

BATCH_SIZE = 128

NBR_EPOCHS = 100

USE_Label_Smoothing = True

ROOT_FOLDER = "/students/julyedu_510477/PersonReID_project"

DATA_FOLDER = '/data/PersonReID/market1501/bounding_box_train'

TEST_DATA_FLDER = '/data/PersonReID/market1501/bounding_box_test'

MobileNet_weight_path = "/students/julyedu_510477/PersonReID_project/MobileNetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_96_no_top.h5"

model_name = "baseline_model_{}.h5".format(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))

model_file_saved = ROOT_FOLDER + "/models/" + model_name
