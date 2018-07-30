"""
Author: Yunpeng Chen
"""
import os
import cv2
import numpy as np

import torch.utils.data as data
import logging


class ImageListIter(data.Dataset):

    def __init__(self, 
                 image_prefix,
                 txt_list,
                 image_transform,
                 name="",
                 force_color=True):
        super(ImageListIter, self).__init__()

        # load image list
        self.image_list = self._get_video_list(txt_list=txt_list)

        # load params
        self.force_color = force_color
        self.image_prefix = image_prefix
        self.image_transform = image_transform
        logging.info("ImageListIter ({:s}) initialized, num: {:d})".format(name,
                      len(self.image_list)))

    def get_image(self, index):
        # get current video info
        im_id, label, img_subpath = self.image_list[index]

        # load image
        image_path = os.path.join(self.image_prefix, img_subpath)
        if self.force_color:
            cv_read_flag = cv2.IMREAD_COLOR
        else:
            cv_read_flag = cv2.IMREAD_GRAYSCALE
        cv_img = cv2.imread(image_path, cv_read_flag)
        image_input = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # apply image augmentation
        if self.image_transform is not None:
            image_input = self.image_transform(image_input)
        return image_input, label, img_subpath


    def __getitem__(self, index):
        image_input, label, img_subpath = self.get_image(index)
        return image_input, label


    def __len__(self):
        return len(self.image_list)


    def _get_video_list(self, txt_list):
        # formate:
        # [im_id, label, image_subpath]
        assert os.path.exists(txt_list), "Failed to locate: {}".format(txt_list)

        # building dataset
        logging.info("Building dataset ...")
        image_list = []
        with open(txt_list) as f:
            lines = f.read().splitlines()
            logging.info("Found {} images in '{}'".format(len(lines), txt_list))
            for i, line in enumerate(lines):
                im_id, label, image_subpath = line.split()
                info = [int(im_id), int(label), image_subpath]
                image_list.append(info)

        return image_list