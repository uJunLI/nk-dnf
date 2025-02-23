#!/usr/bin/env python

import os
import jittor as jt
from jittor import init
from jittor import nn
from jittor.dataset import Dataset

import numpy as np
import rawpy

from glob import glob
import time

from collections.abc import Mapping, Sequence
from PIL import Image
import jittor as jt

from utils.registry import DATASET_REGISTRY
from utils.AverageMeter import AverageMeter


class BaseDictSet(Dataset):

    def __init__(self, data_path, load_npy=True,
                 max_clip=1.0, min_clip=None, ratio=1, **kwargs):
        """
        :param data_path: dataset directory
        :param image_list_file: contains image file names under data_path
        :param patch_size: if None, full images are returned, otherwise patches are returned
        :param split: train or valid
        :param upper: max number of image used for debug
        """
        super().__init__()
        assert os.path.exists(data_path), "data_path: {} not found.".format(data_path)
        self.data_path = data_path
        self.load_npy = load_npy
        self.max_clip = max_clip
        self.min_clip = min_clip
        self.ratio = ratio

        self.raw_short_read_time = AverageMeter()
        self.raw_short_pack_time = AverageMeter()
        self.data_norm_time = AverageMeter()
        self.count = 0

        self.img_info = []
        
        img_list = sorted(glob(f'{self.data_path}/*'))
        for i, img_file in enumerate(img_list):
            img_file = os.path.basename(img_file)
            self.img_info.append({
                'img': img_file,
                'ratio': np.float32(ratio)
            })
        print("processing: {} images".format(len(self.img_info)))

        self.total_len = len(self.img_info)
        self.sampler = None

    def __len__(self):
        return len(self.img_info)

    def print_time(self):
        print('self.raw_short_read_time:', self.raw_short_read_time.avg)
        print('self.raw_short_pack_time:', self.raw_short_pack_time.avg)
        print('self.data_norm_time:', self.data_norm_time.avg)

    def __getitem__(self, index):

        if self.count % 100 == 0 and False:
            self.print_time()
        info = self.img_info[index]

        img_file = info['img']
        if not self.load_npy:
            start = time.time()
            raw = rawpy.imread(os.path.join(self.data_path, img_file))
            self.raw_short_read_time.update(time.time() - start)
            start = time.time()
            noisy_raw = self._pack_raw(raw)
            self.raw_short_pack_time.update(time.time() - start)
        else:
            start = time.time()
            noisy_raw = np.load(os.path.join(self.data_path, img_file), allow_pickle=True)
            self.raw_short_read_time.update(time.time() - start)

        start = time.time()
        noisy_raw = (np.float32(noisy_raw) - self.black_level) / np.float32(self.white_level - self.black_level)  # subtract the black level
        self.data_norm_time.update(time.time() - start)

        if self.ratio:
            noisy_raw = noisy_raw * info['ratio']
        if self.max_clip is not None:
            noisy_raw = np.minimum(noisy_raw, self.max_clip)
        if self.min_clip is not None:
            noisy_raw = np.maximum(noisy_raw, self.min_clip)

        noisy_raw = jt.array(noisy_raw)
        
        self.count += 1
        # 假设 info['ratio'] 是一个 float32 类型的值
        ratio = info['ratio']

        # 判断并转换为 jt.Var 类型
        if isinstance(ratio, np.float32) or isinstance(ratio, float):  # 也可以同时处理 np.float32 和 Python 的 float 类型
            ratio = jt.Var(ratio)  # 转换为 jittor 的 jt.Var

        return {
            'noisy_raw': noisy_raw,
            'img_file': img_file,
            'ratio': ratio
        }



@DATASET_REGISTRY.register()
class SingleBayerDictSet(BaseDictSet):
    def __init__(self, data_path, load_npy=True, max_clip=1, min_clip=None, ratio=1, black_level=512, white_level=16383, **kwargs):
        super().__init__(data_path, load_npy, max_clip, min_clip, ratio, **kwargs)
        self.block_size = 2
        self.black_level = black_level
        self.white_level = white_level

    def _pack_raw(self, raw):
        # pack Bayer image to 4 channels (RGBG)
        im = raw.raw_image_visible.astype(np.uint16)

        H, W = im.shape
        im = np.expand_dims(im, axis=0)
        out = np.concatenate((im[:, 0:H:2, 0:W:2],
                              im[:, 0:H:2, 1:W:2],
                              im[:, 1:H:2, 1:W:2],
                              im[:, 1:H:2, 0:W:2]), axis=0)
        return out
