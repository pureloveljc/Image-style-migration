#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2019-01-13 10:37"

import numpy as np
import tensorflow as tf
vgg_16_data = np.load('./vgg16.npy', encoding='latin1')
# print(vgg_16_data)
# print(type(vgg_16_data))
data_dict = vgg_16_data.item()
print(data_dict.keys())
conv5_1 = data_dict['conv5_1']
w, b = conv5_1
print(w.shape)
print(b.shape)
