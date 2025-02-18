import jittor as jt
from jittor import init
from jittor import nn
from jittor.dataset import DataLoader

import importlib
from copy import deepcopy
from os import path as osp
from glob import glob
from utils.registry import DATASET_REGISTRY
__all__ = ['build_test_loader', 'build_train_loader', 'build_valid_loader']
dataset_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in glob(osp.join(dataset_folder, '*_dataset.py'))]
_dataset_modules = [importlib.import_module(f'datasets.{file_name}') for file_name in dataset_filenames]



def build_dataset(dataset_cfg, split: str):
    assert (split.upper() in ['TRAIN', 'VALID', 'TEST'])
    dataset_cfg = deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop('type')
    process_cfg = dataset_cfg.pop('process')
    split_cfg = dataset_cfg.pop(split)

    # # 检查是否存在 batch_size 配置
    # if 'batch_size' in split_cfg:
    #     batch_size = split_cfg['batch_size']
    # else:
    #     batch_size = 2  # 默认值

    dataset = DATASET_REGISTRY.get(dataset_type)(**dataset_cfg, **process_cfg, **split_cfg, split=split)

    # # 如果 dataset 中没有 batch_size 属性，则强制添加
    # if not hasattr(dataset, 'batch_size'):
    #     setattr(dataset, 'batch_size', batch_size)

    return dataset


def build_train_loader(dataset_cfg):
    train_dataset = build_dataset(dataset_cfg, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=dataset_cfg['train']['batch_size'], shuffle=True, num_workers=dataset_cfg['num_workers'])
    return train_dataloader

def build_valid_loader(dataset_cfg, num_workers=None):
    valid_dataset = build_dataset(dataset_cfg, 'valid')
    if (num_workers is None):
        num_workers = dataset_cfg['num_workers']
    valid_dataloader = DataLoader(valid_dataset, batch_size=dataset_cfg['valid']['batch_size'], shuffle=False, num_workers=num_workers)
    return valid_dataloader

def build_test_loader(dataset_cfg, num_workers=None):
    if num_workers is None:
        num_workers = dataset_cfg['num_workers']
    test_dataset = build_dataset(dataset_cfg, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=dataset_cfg['test']['batch_size'], shuffle=False, num_workers=num_workers)
    return test_dataloader