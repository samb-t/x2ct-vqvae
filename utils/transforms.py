import torch
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage
from scipy import ndimage


# ct transformations:
# taken from:
# https://github.com/kylekma/X2CT
class Resize_image(object):
    '''
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, size=(3,256,256)):
        if not _isArrayLike(size):
            raise ValueError('each dimension of size must be defined')
        self.size = np.array(size, dtype=np.float32)

    def __call__(self, img):
        z, x, y = img.shape
        ori_shape = np.array((z, x, y), dtype=np.float32)
        resize_factor = self.size / ori_shape
        img_copy = ndimage.interpolation.zoom(img, resize_factor, order=1)

        return img_copy


class CPResize_image(object):
    '''
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, size=(3,256,256)):
        if not _isArrayLike(size):
            raise ValueError('each dimension of size must be defined')
        self.size = cp.array(size, dtype=np.float32)

    def __call__(self, img):
        z, x, y = img.shape
        ori_shape = cp.array((z, x, y), dtype=np.float32)
        resize_factor = self.size / ori_shape
        img_copy = cupyx.scipy.ndimage.zoom(img, resize_factor, order=1)

        return img_copy


class Normalization(object):
    '''
    To value range -1 - 1
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, min, max, round_v=6):
        '''
        :param min:
        :param max:
        :param round_v:
          decrease calculating time
        '''
        self.range = np.array((min, max), dtype=np.float32)
        self.round_v = round_v

    def __call__(self, img):
        img_copy = img.copy()
        img_copy = np.round((img_copy - (self.range[0])) / (self.range[1] - self.range[0]), self.round_v)
        # TODO: move this into line above
        img_copy * 2 - 1

        return img_copy


class CPNormalization(object):
    '''
    To value range -1 - 1
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, min, max, round_v=6):
        '''
        :param min:
        :param max:
        :param round_v:
          decrease calculating time
        '''
        self.range = cp.array((min, max), dtype=np.float32)
        self.round_v = round_v

    def __call__(self, img):
        img_copy = img.copy()
        img_copy = cp.round((img_copy - (self.range[0])) / (self.range[1] - self.range[0]), self.round_v)
        img_copy * 2 - 1

        return img_copy


class Normalization_gaussian(object):
    '''
    To value range 0-1
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_copy = img.copy()
        img_copy = (img_copy - self.mean) / self.std

        return img_copy


class Limit_Min_Max_Threshold(object):
    '''
    Restrict in value range. value > max = max,
    value < min = min
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, window_center, window_size):
        self.min = window_center - window_size / 2
        self.max = window_center + window_size / 2

    def __call__(self, img):
        img_copy = img.copy()
        img_copy[img_copy > self.max] = self.max
        img_copy[img_copy < self.min] = self.min
        img_copy = img_copy - self.min

        return img_copy


class ToTensor(object):
    '''
    To Torch Tensor
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __call__(self, img):
        img = torch.from_numpy(img.astype(np.float32))

        return img


class CPToTensor(object):
    '''
    CuPy To Torch Tensor
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __call__(self, img):
        img = torch.as_tensor(img.astype(np.float32), device='cuda')

        return img
    

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


