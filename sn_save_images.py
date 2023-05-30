#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 03:11:32 2020

@author: sr05
"""
import os
from PIL import Image
import sn_config as C


data_path = C.data_path
# pictures_path = C.pictures_path_Source_estimate
pictures_path = os.path.expanduser('~') + '/Python/pictures/Source_Signals/'


image_list = []
path_list = []

for file in os.listdir(pictures_path):
    if file.endswith('.png'):
        path_list.append(file)

path_list.sort()

for image in path_list[1:]:
    image_list.append(Image.open(pictures_path + image).convert('RGB'))

img = Image.open(pictures_path + path_list[0]).convert('RGB')
img.save(pictures_path + 'source_estimation.pdf', save_all=True, append_images=image_list)
