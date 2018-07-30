"""
Author: Yunpeng Chen
"""
import torch
import numpy as np

from .image_transforms import Compose, \
                              Transform, \
                              Normalize, \
                              Resize, \
                              RandomScale, \
                              CenterCrop, \
                              RandomCrop, \
                              RandomHorizontalFlip, \
                              RandomRGB, \
                              RandomHLS


class ToTensor(Transform):
    """Converts a numpy.ndarray (H x W x (T x C)) in the range
    [0, 255] to a torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, clips):
        if isinstance(clips, np.ndarray):
            H, W, _ = clips.shape
            # handle numpy array
            clips = torch.from_numpy(clips.reshape((H,W,-1,self.dim)).transpose((3, 2, 0, 1)))
            # backward compatibility
            return clips.float() / 255.0