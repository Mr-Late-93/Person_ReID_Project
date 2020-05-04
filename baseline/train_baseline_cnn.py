#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/30 9:10
# @Author  : Mr.D
# @Site    :
# @File    : train_baseline_cnn.py
# @Software: PyCharm
"""

baseline:提特征做相似度匹配，在数据集中已知N张图片对应N个label（person_id）.构建一个分类器，最后用一个softmax
         分类，由于测试集是不属于训练集的n个人，可以提取某一卷积层输出的特征图，然后把500张图片也送进网络提取特征，
         最后和N张图片的特征做相似度匹配，再取top10投票选取最终结果

"""
import os
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from utils import generator_batch
from keras.callbacks import TensorBoard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# -----------------------------------------超参设置--------------------------
IMG_WIGTH = 64

IMG_HIGHT = 128

LEARNING_RATE = 0.001

OPTIMIZER = SGD(lr=LEARNING_RATE, momentum=0.9, decay=0.0, nesterov=True)

BATCH_SIZE = 128

NBR_EPOCHS = 100

ROOT_FOLDER = "/students/julyedu_510477/PersonReID_project"

DATA_FOLDER = '/data/PersonReID/market1501/bounding_box_train'

# ----------------------------------------文件名称、路径提取---------------------
image_names = sorted(os.listdir(DATA_FOLDER))

img_name = [x for x in image_names[:-2]]
img_path = [os.path.join(DATA_FOLDER, x) for x in image_names[:-2]]

print("image name top 10 :", img_name[:10])
print("image path top 10 :", img_path[:10])

person_id_original_list = [x[:4] for x in img_name]
nbr_persion_ids = len(person_id_original_list)
print("numbers of persion ids:", nbr_persion_ids)

# ---------------------------------------划分数据集-----------------------------
train_img_path, val_img_path, train_ids, val_ids = train_test_split(img_path, person_id_original_list, test_size=0.2,
                                                                    random_state=2020)
print("numbers of train images:", len(train_img_path))
print("numbers of val images:", len(val_img_path))

# ---------------------------------------backbone------------------------------
weight_path = "/students/julyedu_510477/PersonReID_project/MobileNetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_96_no_top.h5"

backbone = MobileNetV2(weights=weight_path, input_shape=(IMG_HIGHT, IMG_WIGTH, 3), include_top=False, alpha=0.5,
                       pooling='max')

# backbone.summary()

gobal_pool = backbone.get_layer(index=-1).output

dropout_layer = Dropout(0.25)(gobal_pool)

dense = Dense(nbr_persion_ids, activation='softmax')(dropout_layer)

baseline_model = Model(inputs=backbone.input, outputs=dense)

baseline_model.compile(loss='categorical_crossentropy',
                       optimizer=OPTIMIZER,
                       metrics=['acc'])

model_file_saved = ROOT_FOLDER + "/models/baseline_best_MobileNetV2.h5"

checkpoint = ModelCheckpoint(model_file_saved, monitor='val_acc',
                             verbose=1, save_weights_only=True, save_best_only=True, mode='max')

# 学习率衰减
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=5, verbose=1, min_lr=0.00001)

# 早停
early_stop = EarlyStopping(monitor="val_loss", patience=5, verbose=1)

tbCallBack = TensorBoard(log_dir='./logs_2', histogram_freq=0)

train_generator = generator_batch(img_path_list=train_img_path, img_labels=train_ids, nbr_classes=nbr_persion_ids,
                                  batch_size=BATCH_SIZE, shuffle=True, return_label=True, augment=True)

val_generator = generator_batch(img_path_list=val_img_path, img_labels=val_ids, nbr_classes=nbr_persion_ids,
                                batch_size=BATCH_SIZE, shuffle=False, return_label=True, augment=False)

baseline_model.fit_generator(train_generator,
                             steps_per_epoch=len(train_img_path) // BATCH_SIZE,

                             validation_data=val_generator,
                             validation_steps=len(val_img_path) // BATCH_SIZE,
                             verbose=1,
                             shuffle=True,
                             epochs=NBR_EPOCHS,
                             callbacks=[checkpoint, tbCallBack])
