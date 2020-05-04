#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/30 17:21
# @Author  : Mr.D
# @Site    : 
# @File    : utils.py
# @Software: PyCharm


from sklearn.utils import shuffle as shuffle_tuple
from imgaug import augmenters as iaa
import imgaug as ia
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
np.random.seed(1024)


def generator_batch(img_path_list, img_labels, nbr_classes,
                    img_width=64, img_height=128, batch_size=128, shuffle=False,
                    return_label=True, img_resize=False, augment=False):
    """
    generator for train
    :param img_labels:
    :param augment:
    :param img_resize:
    :param return_label:
    :param img_path_list:
    :param nbr_classes:
    :param img_width:
    :param img_height:
    :param batch_size:
    :param shuffle:
    :return:
    """
    N = len(img_path_list)

    if shuffle:
        img_path_list, img_labels = shuffle_tuple(img_path_list, img_labels)

    batch_index = 0

    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        X_batch = np.zeros((current_batch_size, img_height, img_width, 3))
        Y_batch = np.zeros((current_batch_size, nbr_classes))

        for i in range(current_index, current_index + current_batch_size):

            if return_label:
                label = int(img_labels[i])

            img_path = img_path_list[i]

            # image resize
            if img_resize:
                row_img = cv2.imread(img_path)
                img = cv2.resize(row_img, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            else:
                img = cv2.imread(img_path)

            X_batch[i - current_index] = img

            if return_label:
                Y_batch[i - current_index, label] = 1

        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq(images=X_batch)

        # 预处理
        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)

        # img = X_batch[0, :, :, :]
        # img = np.reshape(img, -1)

        # return X_batch, Y_batch

        if return_label:
            yield (X_batch, Y_batch)
        else:
            yield X_batch


# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.85, 1.15), "y": (0.85, 1.5)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},  # translate by -20 to +20 percent (per axis)
            rotate=(-15, 15),  # rotate by -45 to +45 degrees
            shear=(-5, 5),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 3),
                   [
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 2.0)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(1, 5)),  # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(1, 5)),  # blur image using local medians with kernel sizes between 2 and 7
                       ]),

                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03 * 255), per_channel=0.5),
                       # add gaussian noise to images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           iaa.CoarseDropout((0.03, 0.15), size_percent=(0.01, 0.03), per_channel=0.2),
                       ]),
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)

                       iaa.ContrastNormalization((0.3, 1.0), per_channel=0.5),  # improve or worsen the contrast
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                       # move pixels locally around (with random strengths)
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)
