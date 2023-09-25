import torch
import numpy as np
from scipy import ndimage


class Resize_image(object):
    '''
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, size=(3,256,256), f16=False):
        if not _isArrayLike(size):
            raise ValueError('each dimension of size must be defined')
        if f16:
            self.size = np.array(size, dtype=np.float16)
        else:
            self.size = np.array(size, dtype=np.float32)
        self.f16 = f16

    def __call__(self, img):
        z, x, y = img.shape
        assert not self.f16, "Resize_image not supported for f16"
        ori_shape = np.array((z, x, y), dtype=np.float32)
        resize_factor = self.size / ori_shape
        return ndimage.interpolation.zoom(img, resize_factor, order=1)


class Normalization(object):
    '''
    To value range -1 - 1
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, min, max, f16=False, round_v=6):
        '''
        :param min:
        :param max:
        :param f16:
          use float16 for mixed precision
        :param round_v:
          decrease calculating time
        '''
        dtype = np.float16 if f16 else np.float32
        self.range = np.array((min, max), dtype=dtype)
        self.round_v = round_v

    def __call__(self, img):
        img_copy = img.copy()
        img_copy = np.round((img_copy - (self.range[0])) / (self.range[1] - self.range[0]), self.round_v)

        return img_copy


class Normalization_min_max(object):
    '''
    To value range min, max
    img: 3D, (z, y, x) or (D, H, W)
    remove_noise: Set true for baggage data
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, min_v, max_v, eps=1e-4, remove_noise=True):
        self.max = max_v
        self.min = min_v
        self.eps = eps
        self.remove_noise = remove_noise

    def __call__(self, img):
        # Removing noise from xray machine
        if self.remove_noise:
            img[img < 200] = 0
        img_min = np.min(img)
        img_max = np.max(img)

        img_out = (self.max - self.min) * (img - img_min) / (img_max - img_min + self.eps) + self.min
        return img_out


class ReturnIdentity(object):
    def __call__(self, img):
        return img


class ToTensor(object):
    '''
    To Torch Tensor
    img: 3D, (z, y, x) or (D, H, W)
    :param f16:
      use float16 for mixed precision
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, f16=False):
        self.f16 = f16

    def __call__(self, img):
        dtype = np.float16 if self.f16 else np.float32
        return torch.from_numpy(img.astype(dtype))
    

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
