# import _init_paths
import os
# import cv2
import torch
import torch.utils.data
# import random
import argparse
import numpy as np

import lib.models.models as models

from os.path import exists, join, dirname, realpath
from easydict import EasyDict as edict
from lib.utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

from lib.eval_toolkit.pysot.datasets import VOTDataset
from lib.eval_toolkit.pysot.evaluation import EAOBenchmark
from lib.tracker.lighttrack import Lighttrack


def parse_args():
    parser = argparse.ArgumentParser(description='Test LightTrack')
    parser.add_argument('--arch', dest='arch', help='backbone architecture')
    parser.add_argument('--resume', type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2019', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    parser.add_argument('--video', default=None, type=str, help='test a video in benchmark')
    parser.add_argument('--stride', type=int, help='network stride')
    parser.add_argument('--even', type=int, default=0)
    parser.add_argument('--path_name', type=str, default='NULL')
    args = parser.parse_args()

    return args


def load_checkpoint(model,
                    load_path,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file.

    Args:
        model (Module): Module to load checkpoint.
        load_path (str): a file path.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if not os.path.isfile(load_path):
        raise IOError('{} is not a checkpoint file'.format(load_path))

    checkpoint = torch.load(load_path, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(load_path))

    # strip prefix of state_dict
    # when using DataParallel, the saved names have prefix 'module.'
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # load state_dict
    assert not hasattr(model, 'module'), 'do not use DataParallel to wrap the model before loading state dict!'
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if rank == 0:
        if missing_keys:
            if logger is not None:
                logger.info('[state dict loading warning] missing keys: {}'.format(','.join(missing_keys)))
            else:
                print('[state dict loading warning] missing keys: {}'.format(','.join(missing_keys)))

        if unexpected_keys:
            if logger is not None:
                logger.info('[state dict loading warning] unexpected keys: {}'.format(','.join(missing_keys)))
            else:
                print('[state dict loading warning] unexpected keys: {}'.format(','.join(missing_keys)))

    return model


def main():
    args = parse_args()

    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.epoch_test = args.epoch_test
    info.stride = args.stride

    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = args.dataset
    siam_info.epoch_test = args.epoch_test
    siam_info.stride = args.stride
    # build tracker
    siam_tracker = Lighttrack(siam_info, even=args.even)
    # build siamese network
    if args.path_name != 'NULL':
        siam_net = models.__dict__[args.arch](args.path_name, stride=siam_info.stride)
    else:
        siam_net = models.__dict__[args.arch](stride=siam_info.stride)

    print('===> init Siamese <====')

    siam_net = load_pretrain(siam_net, args.resume)
    siam_net.eval()
    print(siam_net)
    torch.save(siam_net, "ligthtrack.pt")


    # convert init-net
    x = torch.randn(1, 3, 127, 127)
    input_names = ["input1"]
    output_names = ["output.1"]
    # fix shape
    torch.onnx.export(siam_net, x, "lighttrack_init.onnx", verbose=True, input_names=input_names,
                      output_names=output_names)
    #
    # # convert backone-net
    # x = torch.randn(1, 3, 288, 288)
    # input_names = ["input1"]
    # output_names = ["output.1"]
    # # fix shape
    # torch.onnx.export(siam_net, x, "lighttrack_backbone.onnx", verbose=True, input_names=input_names,
    #                   output_names=output_names)


    # convert track
    # zf = torch.randn(1, 96, 8, 8)
    # xf = torch.randn(1, 96, 18, 18)
    # input_names = ["input1"] + ["input2"]
    # output_names = ["output.1"] + ["output.2"]
    # # fix shape
    # torch.onnx.export(siam_net, (zf, xf), "lighttrack_neck_head.onnx", verbose=True, input_names=input_names, output_names=output_names)


if __name__ == '__main__':
    main()
