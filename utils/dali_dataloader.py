import os
import glob
import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from utils.transforms import *
from nvidia import dali


class XCT_dataset(Dataset):
    """
    Class for loading CT scans
    and paired Digitaly Reconstructed Radiographs
    (DRR) images
    """
    def __init__(self, data_dir, train, xray_scale=256, ct_scale=128, projections=2,
                 ct_min=0, ct_max=2000, scale_ct=True, load_res=None):
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
        split_file = 'data/train_split.txt' if train else 'data/test_split.txt'
        self.projections = projections
        self.data_dir = data_dir
        self.ct_min = ct_min
        self.load_res = load_res

        if projections == 2:
            self.view_list = [0,2]
        elif projections == 3:
            self.view_list = [0,1,2]
        elif projections == 4:
            self.view_list = [0,2,4,6]
        else:
            self.view_list = [0,1,2,3,4,5,6,7]

        self.split_list = self._get_split(split_file)
        self.object_dirs = self._get_dirs()
        
        self.xray_tx = [torchvision.transforms.Resize(xray_scale),
                        Normalization(0, 255),
                        # Removed gaussian normalization since with 0, 1 it's the identity 
                        # Normalization_gaussian(0., 1),
                        ToTensor()]

        self.ct_tx = [Resize_image((ct_scale, ct_scale, ct_scale)) if scale_ct else nn.Identity(),
                      Limit_Min_Max_Threshold(ct_min, ct_max),
                      Normalization(ct_min, ct_max), 
                      # Removed gaussian normalization since with 0, 1 it's the identity 
                      # Normalization_gaussian(0., 1.),
                      ToTensor()]

    def __len__(self):
        return len(self.object_dirs)

    def __getitem__(self, idx):
        object_dir = self.object_dirs[idx]
        
        xray_files = sorted(glob.glob(f'{object_dir}/*.png'))
        xrays_list = []
        for i in self.view_list:
            xray = Image.open(xray_files[i])
            for transf in self.xray_tx:
                xray = transf(xray)
            xrays_list.append(xray)
        xrays = torch.stack(xrays_list, 0)

        if self.load_res is not None:
            ct_path = f'{object_dir}/*_{self.load_res}.npz'
            ct_file = glob.glob(ct_path)
            ct_scan = torch.from_numpy(np.load(ct_file[0])["ct"]).squeeze(0)
        else:
            ct_path = f'{object_dir}/*.npy'
            ct_file = glob.glob(ct_path)
            ct_scan = torch.from_numpy(np.load(ct_file[0]))

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
        data_dirs = glob.glob(f"{self.data_dir}/**/*_chest", recursive=True)
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
    def __init__(self, data_dir, train, scale=128, projections=2):
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
        split_file = 'data/train_split.txt' if train else 'data/test_split.txt'
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

        self.split_list = self._get_split(split_file)
        self.xrays_files = self._get_files()
        
        
        self.xray_tx = [torchvision.transforms.Resize(scale),
                        Normalization(0, 255),
                        # Removed gaussian normalization since with 0, 1 it's the identity 
                        # Normalization_gaussian(0., 1),
                        ToTensor()]

    def __len__(self):
        return len(self.xrays_files)

    def __getitem__(self, idx):
        xray_file = self.xrays_files[idx]
        xray = Image.open(xray_file)
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
            xray_files = sorted(glob.glob(f'{object_dir}/*.png'))
            for i in self.view_list:
                xray = xray_files[i]
                files.append(xray)
        return files
    
    def _get_dirs(self):
        data_dirs = glob.glob(f"{self.data_dir}/**/*_chest", recursive=True)
        dirs = [x for x in data_dirs if os.path.basename(x).split('_')[0] in set(self.split_list)]
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
                 ct_min=0, ct_max=2000, scale_ct=True, use_dali=True,
                 dataset='chest'):
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
        self.dataset = dataset
        split_file = f'data/{self.dataset}_train_split.txt' if train else f'data/{self.dataset}_test_split.txt'
        self.data_dir = data_dir
        self.ct_min = ct_min
        self.load_res = load_res
        self.idx = None

        self.split_list = self._get_split(split_file)
        self.object_dirs = self._get_dirs()

        self.ct_tx = [Resize_image((scale, scale, scale)) if scale_ct else nn.Identity(),
                      Limit_Min_Max_Threshold(ct_min, ct_max),
                      Normalization(ct_min, ct_max), 
                      # Removed gaussian normalization since with 0, 1 it's the identity 
                      # Normalization_gaussian(0., 1.),
                      ToTensor()]
        
        if self.use_dali:
            self.augmentations = {}
            init_dali_augmentations()

    def init_dali_augmentations(self):
        self.augmentations["resize"] = \
            lambda vol: dali.fn.resize(vol, resize_z=224, mode="default", interp_type=dali.types.INTERP_LANCZOS3)
        angle = dali.fn.random.uniform(range=(-20, 20), seed=123)
        axis = dali.fn.random.uniform(range=(-1,1), shape=[3])
        self.augmentations["rotate"] = \
            lambda vol: dali.fn.rotate(vol, angle=angle, axis=axis, interp_type=dali.types.INTERP_LINEAR, fill_value=0)

    def run(p):
        p.build() # builds the dali pipeline
        return p.run()

    @dali.pipeline_def
    def pipe(self):
        # defines dataloading pipeline for nvidia dali
        # gpu device only works if GPUDirect Storage Support is installed
        # else use cpu or set use_dali=False
        data = dali.fn.readers.numpy(device='gpu',
                                     file_root=self.objects_dirs[self.idx],
                                     filer_filter='*.npy')
        for aug in self.augmentations.values():
            data = aug(data)
        return data

    def __len__(self):
        return len(self.object_dirs)

    def __getitem__(self, idx):
        self.idx = idx
        if self.use_dali:
            pipe = self.pipe()
            data = {
                "ct": self.run(pipe),
                "dir_name": self.object_dirs[idx]}
            return data
            
        object_dir = self.object_dirs[idx]

        if self.load_res is not None:
            ct_path = f'{object_dir}/*_{self.load_res}.npz'
            ct_file = glob.glob(ct_path)
            ct_scan = torch.from_numpy(np.load(ct_file[0])["ct"]).squeeze(0)
        else:
            ct_path = f'{object_dir}/*.npy'
            ct_file = glob.glob(ct_path)
            ct_scan = torch.from_numpy(np.load(ct_file[0]))

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
        

def get_data(data_dir, train, img_size, num_xrays,
             ct_min, ct_max):
    assert os.path.isdir(data_dir)
        
    dataset = XCT_dataset(data_dir, train, scale=img_size,
                          projections=num_xrays,
                          ct_min=ct_min, ct_max=ct_max)

    return dataset

# Usage example:
'''
from torch.utils.data.dataloader import DataLoader
device = torch.device("cuda:0")
dataset = get_data(data_dir='/projects/cgw/medical/lidc',
                  train=True, img_size=128,
                  num_xrays=2, ct_min=0, ct_max=2000)
train_loader = DataLoader(dataset,
                          batch_size=1,
                          shuffle=True)

data_loader = iter(train_loader)
data = next(data_loader)
xrays = data['xrays'].to(device)
ct = data['ct'].to(device)
path = data['dir_name']
'''
