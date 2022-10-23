import os
import glob
import torch
import torch.nn as nn
import numpy as np
import cupy as cp
import torchvision
from random import random
from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from utils.transforms import *


class XCT_dataset(Dataset):
    """
    Class for loading CT scans
    and paired Digitaly Reconstructed Radiographs
    (DRR) images
    """
    def __init__(self, data_dir, train, dataset='chest', xray_scale=256, ct_scale=128, projections=2,
                 ct_min=0, ct_max=2000, scale_ct=True, load_res=None, cupy=True, aug_prob=0.2,
                 use_synthetic=True):
        """
        :param data_dir: (str) path to folder containing all ct scans

        :param train: (bool) are we training or testing.
        Data splits taken from https://openaccess.thecvf.com/content_CVPR_2019/html/Ying_X2CT-GAN_Reconstructing_CT_From_Biplanar_X-Rays_With_Generative_Adversarial_Networks_CVPR_2019_paper.html

        :param scale: (int) resize both x-ray images and ct volumes to this value

        :param projections: (int) how many xray projections to use
        these are stacked along the 0 dim (projections, scale, scale)
        3 projections (sagittal, one at 45 deg and coronal)
        2 projections (sagittal & coronal)
        4 projections (one every 90 degrees)
        8 projections (one every 45 degrees)

        :param ct_min, ct_max: (int)  min and max values to adjust the intensities. 
        These values are taken from the preprocessing of the scans.
        should only be changed if the data preprocessing changes

        :return data: (dict)
        'xrays': stacked x-ray projections torch.Tensor(projections, scale, scale)
        'ct': ct volume torch.Tensor(scale, scale, scale)
        'dir_name': full path to that ct scan (str)
        """
        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.dataset = dataset
        self.ext = '*jpeg' if use_synthetic and dataset=='knee' else '*.png'
        split_file = f'data/{self.dataset}_train_split.txt' if train else f'data/{self.dataset}_test_split.txt'
        self.projections = projections
        self.data_dir = data_dir
        self.ct_min = ct_min
        self.cupy = cupy
        self.load_res = load_res

        if projections == 2:
            self.view_list = ([0,2])
        elif projections == 3:
            self.view_list = ([0,1,2])
        elif projections == 4:
            self.view_list = ([0,2,4,6])
        else:
            self.view_list = ([0,1,2,3,4,5,6,7])
        if dataset == 'knee' and not use_synthetic:
            self.view_list = [0, 1]

        self.split_list = self._get_split(split_file)
        self.object_dirs = self._get_dirs()
        
        self.xray_tx = [torchvision.transforms.Resize(xray_scale),
                        Normalization(0, 255),
                        ToTensor()]
        if cupy:
            self.ct_tx = [Resize_image((ct_scale, ct_scale, ct_scale), cupy=True),
                          Limit_Min_Max_Threshold(ct_min, ct_max),
                          Normalization(ct_min, ct_max, cupy=True),
                          ToTensor(cupy=True)]
        else:
            self.ct_tx = [Resize_image((ct_scale, ct_scale, ct_scale)) if scale_ct else nn.Identity(),
                          Limit_Min_Max_Threshold(ct_min, ct_max),
                          Normalization(ct_min, ct_max), 
                          ToTensor()]

    def __len__(self):
        return len(self.object_dirs)

    def __getitem__(self, idx):
        object_dir = self.object_dirs[idx]
        xray_files = sorted(glob.glob(os.path.join(object_dir, self.ext)))
        xrays_list = []
        for i in self.view_list:
            xray = Image.open(xray_files[i]).convert('L')
            for transf in self.xray_tx:
                xray = transf(xray)
            xrays_list.append(xray)
        xrays = torch.stack(xrays_list, 0)

        if self.load_res is not None:
            ct_path = f'{object_dir}/*_{self.load_res}.npz'
            ct_file = glob.glob(ct_path)
            ct_scan = cp.squeeze(cp.load(ct_file[0])["ct"], 0) if self.cupy else torch.from_numpy(np.load(ct_file[0])["ct"]).squeeze(0)
        else:
            ct_file = glob.glob(os.path.join(object_dir, '*.npy'))
            ct_scan = cp.load(ct_file[0]) if self.cupy else torch.from_numpy(np.load(ct_file[0]))

        for transf in self.ct_tx:
            ct_scan = transf(ct_scan)
        
        xrays = xrays.unsqueeze(1)
        ct_scan = ct_scan.unsqueeze(0)

        data = {
            "xrays": xrays,
            "ct": ct_scan,
            "dir_name": object_dir}

        return data

    def _get_dirs(self):
        data_dirs = glob.glob(f"{self.data_dir}/**/*_{self.dataset}", recursive=True)
        dirs = [x for x in data_dirs if os.path.basename(x).split('_')[0] in set(self.split_list)]
        return dirs

    @staticmethod
    def _get_split(split_file):
        with open(split_file) as file:
            data = [n.rstrip() for n in file]
        return data


class XRay_dataset(Dataset):
    """
    Class for loading DRRs
    """
    def __init__(self, data_dir, train, scale=128, projections=2, dataset='chest', aug_prob=0.,
                 use_synthetic=False):
        """
        :param data_dir: (str) path to folder containing all ct scans

        :param train: (bool) are we training or testing.
        Data splits taken from https://openaccess.thecvf.com/content_CVPR_2019/html/Ying_X2CT-GAN_Reconstructing_CT_From_Biplanar_X-Rays_With_Generative_Adversarial_Networks_CVPR_2019_paper.html

        :param scale: (int) resize both x-ray images and ct volumes to this value

        :param projections: (int) how many xray projections to use
        these are stacked along the 0 dim (projections, scale, scale)
        3 projections (sagittal, one at 45 deg and coronal)
        2 projections (sagittal & coronal)
        4 projections (one every 90 degrees)
        8 projections (one every 45 degrees)

        :return data: (dict)
        'xrays': stacked x-ray projections torch.Tensor(projections, scale, scale)
        'ct': ct volume torch.Tensor(scale, scale, scale)
        'dir_name': full path to that ct scan (str)
        """
        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.dataset = dataset
        self.types = ['*.png', '*jpeg'] if use_synthetic else ['*.png']
        split_file = f'data/{self.dataset}_train_split.txt' if train else f'data/{self.dataset}_test_split.txt'
        self.projections = projections
        self.data_dir = data_dir

        if projections == 2:
            self.view_list = [0,2]
        elif projections == 3:
            self.view_list = [0,1,2]
        elif projections == 4:
            self.view_list = [0,2,4,6]
        else:
            self.view_list = [0,1,2,3,4,5,6,7]
        if self.dataset == 'knee':
            self.view_list = [0, 1]

        self.split_list = self._get_split(split_file)
        self.xrays_files = self._get_files()
        
        self.xray_tx = [transforms.Resize(scale),
                        transforms.RandomApply(torch.nn.ModuleList([
                            transforms.RandomResizedCrop(scale, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                            transforms.CenterCrop(scale)]),
                                               p=aug_prob),
                        Normalization(0, 255),
                        ToTensor()]

    def __len__(self):
        return len(self.xrays_files)

    def __getitem__(self, idx):
        xray_file = self.xrays_files[idx]
        xray = Image.open(xray_file).convert('L')
        for transf in self.xray_tx:
            xray = transf(xray)
        
        data = {
            "xray": xray.unsqueeze(0) # add channels dim
        }

        return data

    def _get_files(self):
        dirs = self._get_dirs()
        files = []
            
        for object_dir in dirs:
            for ext in self.types:
                xray_files = sorted(glob.glob(os.path.join(object_dir, ext)))
                for i in self.view_list:
                    xray = xray_files[i]
                    files.append(xray)
        return files
    
    def _get_dirs(self):
        data_dirs = glob.glob(f"{self.data_dir}/**/*_{self.dataset}", recursive=True)
        dirs = [x for x in data_dirs if os.path.basename(x).split('_')[0] in set(self.split_list)]
        return dirs

    @staticmethod
    def _get_split(split_file):
        with open(split_file) as file:
            data = [n.rstrip() for n in file]
        return data


class BagXRay_dataset(Dataset):
    """
    Class for loading Xray views from ct baggage
    """
    def __init__(self,
                 data_dir,
                 train,
                 scale=128,
                 types='grayscale',
                 direction='both',
                 degrees=45,
    ):
        """
        :param data_dir: (str) path to folder containing all ct scans
        :param train: (bool) are we training or testing.
        :param scale: (int) resize x-ray images to this value
        :param types: (str) ['grayscale', 'rgb'] rgb uses the color transfer function of the xray machine, while grayscale the original density values
        :param direction: (str) ['azimuth', 'elevation', 'both'] camera rotation of xrays
        :param degrees: (int) [45, 90, 180] difference between neighboring angles
        """

        self.data_dir = data_dir
        self.types = types
        self.direction = direction
        self.degrees = degrees
        split_file = 'data/bags_train_split.txt' if train else 'data/bags_test_split.txt'
        self.split_list = self._get_split(split_file)
        self.xrays_files = self._get_files()
        
        self.xray_tx = [transforms.Resize(scale),
                        Normalization(0, 255),
                        ToTensor()]

    def __len__(self):
        return len(self.xrays_files)

    def __getitem__(self, idx):
        xray_file = self.xrays_files[idx]
        xray = Image.open(xray_file)
        xray = xray.convert('L') if self.types == 'grayscale' else xray.convert('RGB')
        for transf in self.xray_tx:
            xray = transf(xray)

        if self.types == 'grayscale':
            xray = xray.unsqueeze(0)
        
        data = {
            "xray": xray
        }

        return data


    def _get_dirs(self):
        data_dirs =  [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        return [x for x in data_dirs if os.path.basename(x) in set(self.split_list)]

    def _get_files(self):
        dirs = self._get_dirs()
        files = []
        channels = ['rgb', 'grayscale'] if self.types == 'both' else [self.types]
        camera = ['azimuth', 'elevation'] if self.direction == 'both' else [self.direction]

        for d in dirs:
            for ch in channels:
                for cam in camera:
                    for deg in range(0, 315+1, self.degrees):
                        file_name = f"projections_{ch}/{os.path.basename(d)}_{cam}_{deg}.png"
                        files.append(os.path.join(d, file_name))

        return files

    @staticmethod
    def _get_split(split_file):
        with open(split_file) as file:
            data = [n.rstrip() for n in file]
        return data


class BagCT_dataset(Dataset):
    """
    Class for loading CT scans
    Note: if using cupy to read into gpu, workers need to be 0
    and pin_memory=False when creating dataloader
    """
    def __init__(self, data_dir, train, load_res=None, scale=256,
                 ct_min=0, ct_max=2000, scale_ct=True,
                 cupy=False):
    
        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.cupy = cupy
        split_file = 'data/bags_train_split.txt' if train else 'data/bags_test_split.txt'
        self.data_dir = data_dir
        self.ct_min = ct_min
        self.load_res = load_res

        self.split_list = self._get_split(split_file)
        self.object_dirs = self._get_dirs()

        if cupy:
            self.ct_tx = [Resize_image((scale, scale, scale), cupy=True),
                          Normalization_min_max(0., 1., cupy=True),
                          ToTensor(cupy=True)]
        else:
            self.ct_tx = [Resize_image((scale, scale, scale)) if scale_ct else nn.Identity(),
                          Normalization_min_max(0., 1., cupy=False),
                          ToTensor()]


    def __len__(self):
        return len(self.object_dirs)

    def __getitem__(self, idx):
        object_dir = self.object_dirs[idx]
        ct_path = f'{object_dir}/npz/*.npy'
        ct_file = glob.glob(ct_path)
        ct_scan = cp.load(ct_file[0]) if self.cupy else torch.from_numpy(np.load(ct_file[0]))

        for transf in self.ct_tx:
            ct_scan = transf(ct_scan)

        ct_scan = ct_scan.unsqueeze(0)
        
        data = {
            "ct": ct_scan,
            "dir_name": object_dir}

        return data

    def _get_dirs(self):
        data_dirs = os.listdir(self.data_dir)
        dirs = [os.path.join(self.data_dir, x) for x in data_dirs if x in set(self.split_list)]
        return dirs

    @staticmethod
    def _get_split(split_file):
        with open(split_file) as file:
            data = [n.rstrip() for n in file]
        return data


class CT_dataset(Dataset):
    """
    Class for loading CT scans
    """
    def __init__(self, data_dir, train, load_res=None, scale=256,
                 ct_min=0, ct_max=2000, scale_ct=True, dataset='chest',
                 cupy=False):
        """
        :param data_dir: (str) path to folder containing all ct scans

        :param train: (bool) are we training or testing.
        Data splits taken from https://openaccess.thecvf.com/content_CVPR_2019/html/Ying_X2CT-GAN_Reconstructing_CT_From_Biplanar_X-Rays_With_Generative_Adversarial_Networks_CVPR_2019_paper.html

        :param scale: (int) resize both x-ray images and ct volumes to this value

        :param projections: (int) how many xray projections to use
        these are stacked along the 0 dim (projections, scale, scale)
        3 projections (sagittal, one at 45 deg and coronal)
        2 projections (sagittal & coronal)
        4 projections (one every 90 degrees)
        8 projections (one every 45 degrees)

        :param ct_min, ct_max: (int)  min and max values to adjust the intensities. 
        These values are taken from the preprocessing of the scans.
        should only be changed if the data preprocessing changes

        :return data: (dict)
        'xrays': stacked x-ray projections torch.Tensor(projections, scale, scale)
        'ct': ct volume torch.Tensor(scale, scale, scale)
        'dir_name': full path to that ct scan (str)
        """
        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.dataset = dataset
        self.cupy = cupy
        split_file = f'data/{self.dataset}_train_split.txt' if train else f'data/{self.dataset}_test_split.txt'
        self.data_dir = data_dir
        self.ct_min = ct_min
        self.load_res = load_res

        self.split_list = self._get_split(split_file)
        self.object_dirs = self._get_dirs()

        if cupy:
            self.ct_tx = [Resize_image((scale, scale, scale), cupy=True),
                          Limit_Min_Max_Threshold(ct_min, ct_max),
                          Normalization(ct_min, ct_max, cupy=True),
                          ToTensor(cupy=True)]
        else:
            self.ct_tx = [Resize_image((scale, scale, scale)) if scale_ct else nn.Identity(),
                          Limit_Min_Max_Threshold(ct_min, ct_max),
                          Normalization(ct_min, ct_max), 
                          ToTensor()]


    def __len__(self):
        return len(self.object_dirs)

    def __getitem__(self, idx):
        object_dir = self.object_dirs[idx]

        if self.load_res is not None:
            ct_path = f'{object_dir}/*_{self.load_res}.npz'
            ct_file = glob.glob(ct_path)
            ct_scan = cp.squeeze(cp.load(ct_file[0])["ct"], 0) if self.cupy else torch.from_numpy(np.load(ct_file[0])["ct"]).squeeze(0)
        else:
            ct_path = f'{object_dir}/*.npy'
            ct_file = glob.glob(ct_path)
            ct_scan = cp.load(ct_file[0]) if self.cupy else torch.from_numpy(np.load(ct_file[0]))

        for transf in self.ct_tx:
            ct_scan = transf(ct_scan)

        ct_scan = ct_scan.unsqueeze(0)
        
        data = {
            "ct": ct_scan,
            "dir_name": object_dir}

        return data

    def _get_dirs(self):
        data_dirs = glob.glob(f"{self.data_dir}/**/*_{self.dataset}", recursive=True)
        dirs = [x for x in data_dirs if os.path.basename(x).split('_')[0] in set(self.split_list)]
        return dirs

    @staticmethod
    def _get_split(split_file):
        with open(split_file) as file:
            data = [n.rstrip() for n in file]
        return data
