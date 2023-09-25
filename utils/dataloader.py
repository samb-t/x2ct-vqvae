import os
import glob
import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from utils.transforms import *
from pathlib import Path
from scipy.ndimage import gaussian_filter


class XCT_dataset(Dataset):
    """
    Class for loading chest CT scans
    and paired (for evaluation) Digitaly Reconstructed Radiographs
    (DRR) images
    """
    def __init__(self,
                 data_dir,
                 train,
                 dataset='chest',
                 xray_scale=256,
                 scale=128,
                 projections=2,
                 scale_ct=False,
                 load_res=None,
                 f16=False,
                 use_synthetic=True):
        """
        :param data_dir: (str) path to data folder
        :param train: (bool) are we training or testing.
        :param scale: (int) resize both x-ray images and ct volumes to this value

        :param projections: (int) how many xray projections to use
          3 projections (sagittal, one at 45 deg and coronal)
          2 projections (sagittal & coronal)
          4 projections (one every 90 degrees)
          8 projections (one every 45 degrees)

        :param ct_min, ct_max: (int)  min and max values to adjust the intensities. 
        These values are taken from the preprocessing of the scans.
        should only be changed if the data preprocessing changes

        :return data: (dict)
        'xrays': stacked x-ray projections torch.Tensor(projections, 1, scale, scale)
        'ct': ct volume torch.Tensor(1, scale, scale, scale)
        'dir_name': full path to that ct scan (str)
        """
        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.dataset = dataset
        self.ext = '*jpeg' if use_synthetic and dataset=='knee' else '*.png'
        split_file = f'data/{self.dataset}_train_split.txt' if train else f'data/{self.dataset}_test_split.txt'
        self.projections = projections
        self.data_dir = data_dir
        self.f16 = f16
        self.load_res = load_res

        view_lists = {
            2: [0, 2],
            3: [0, 1, 2],
            4: [0, 2, 4, 6],
        }
        self.view_list = view_lists.get(projections, [0, 1, 2, 3, 4, 5, 6, 7])
        if dataset == 'knee' and not use_synthetic:
            self.view_list = [0, 1]

        self.split_list = self._get_split(split_file)
        self.object_dirs = self._get_dirs()
        
        self.xray_tx = [torchvision.transforms.Resize(xray_scale),
                        Normalization(0, 255),
                        ToTensor()]

        self.ct_tx = [Resize_image((scale, scale, scale)) if scale_ct else nn.Identity(),
                      Normalization_min_max(0., 1., remove_noise=False), 
                      ToTensor(f16=self.f16)]

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
            ct_scan = torch.from_numpy(np.load(ct_file[0])["ct"]).squeeze(0)
        else:
            pattern = f'*{self.dataset}_f16.npz' if self.f16 else f'*{self.dataset}128_f16.npz'
            ct_path = f'{object_dir}/{pattern}'
            ct_file = glob.glob(ct_path)
            ct_scan = np.flip(np.load(ct_file[0])['ct'])

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
    def __init__(self, data_dir, train, scale=128, projections=2,
                 dataset='chest', use_synthetic=False):
        """
        :param data_dir: (str) path to folder containing all ct scans
        :param train: (bool) are we training or testing.
        :param scale: (int) resize both x-ray images and ct volumes to this value
        :param projections: (int) how many xray projections to use
          3 projections (sagittal, one at 45 deg and coronal)
          2 projections (sagittal & coronal)
          4 projections (one every 90 degrees)
          8 projections (one every 45 degrees)
        :param dataset: chest/knee
        :param use_synthetic: use real or synthetic xrays? (only for knee dataset)

        :return data: (dict)
        'xrays': torch.Tensor(1, scale, scale)
        """
        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.dataset = dataset
        self.types = ['*.jpeg'] if use_synthetic else ['*.png']
        split_file = f'data/{self.dataset}_train_split.txt' if train else f'data/{self.dataset}_test_split.txt'
        self.projections = projections
        self.data_dir = data_dir

        view_lists = {
            2: [0, 2],
            3: [0, 1, 2],
            4: [0, 2, 4, 6],
        }
        self.view_list = view_lists.get(projections, [0, 1, 2, 3, 4, 5, 6, 7])
        if dataset == 'knee' and not use_synthetic:
            self.view_list = [0, 1]

        self.split_list = self._get_split(split_file)
        self.xrays_files = self._get_files()
        
        self.xray_tx = [transforms.Resize(scale),
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


class BagXCT_dataset(Dataset):
    """
    Class for loading CT scans
    and paired (for evaluation) Digitaly Reconstructed Radiographs
    (DRR) images from the baggage security dataset
    """
    def __init__(self, data_dir, train, xray_scale=256, ct_scale=128,
                 direction='both', scale_ct=False,
                 types='grayscale', split_shuffle=True,
                 split_seed=500, use_f16=True):
        """
        :param data_dir: (str) path to data folder
        :param train: (bool) are we training or testing.
        :param xray_scale: (int) resize xrays (use the same in xray config)
        :param ct_scale: (int) resize ct (use the same in ct config) (resize does not applies for float16)
        :param direction: (str) ['azimuth', 'elevation', 'both'] direction of projections
        :param scale_ct: (bool) want to scale the ct volume? else return identity
        :param types: (str) ['grayscale', 'rgb']
        :param split_shuffle: (bool) shuffle the train and test splits
        :param split_seed: (int) random seed for reproducing results
        :param use_f16: (bool) load f16 ct volume to save memory

        :return data: (dict)
        'xrays': stacked x-ray projections torch.Tensor(projections, scale, scale)
        'ct': ct volume torch.Tensor(scale, scale, scale)
        'dir_name': full path to that ct scan (str)
        """

        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.train = train
        self.data_dir = data_dir
        self.types = types
        self.direction = direction
        self.scale_ct = scale_ct
        self.seed = split_seed
        self.shuffle = split_shuffle
        self.f16 = use_f16

        self.split_dirs = self._get_dirs()
        
        self.xray_tx = [transforms.Resize(xray_scale),
                        Normalization(0, 255),
                        ToTensor()]

        self.ct_tx = [Resize_image((ct_scale, ct_scale, ct_scale), f16=self.f16) if scale_ct else ReturnIdentity(),
                      Normalization_min_max(0., 1.),
                      ToTensor(f16=self.f16)]


    def __len__(self):
        return len(self.split_dirs)

    def __getitem__(self, idx):
        object_dir = os.path.dirname(os.path.dirname(self.split_dirs[idx]))
        ct_file = self.split_dirs[idx]
        xrays_list = []
        
        channels = ['rgb', 'grayscale'] if self.types == 'both' else [self.types]
        camera = ['azimuth', 'elevation'] if self.direction == 'both' else [self.direction]
        for ch in channels:
            for cam in camera:
                file_name = f"projections_{ch}2/{os.path.basename(object_dir)}_{cam}_{90}.png"
                xray = Image.open(os.path.join(object_dir, file_name))
                xray = xray.convert('L') if self.types == 'grayscale' else xray.convert('RGB')
                for transf in self.xray_tx:
                    xray = transf(xray)
                xrays_list.append(xray)
        xrays = torch.stack(xrays_list, 0)

        ct_scan = np.load(ct_file)['ct'] #if self.f16 else np.load(ct_file)

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
        data_dirs = [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        data_dirs.sort()
        files = []

        # pattern = '*_f16.npz' if self.f16 else '*.npy'

        for folder in data_dirs:
            file_list = list(Path(folder).rglob(pattern))
            if len(file_list) > 0 and "projections_grayscale2" in set(os.listdir(folder)):
                files.append(file_list[0])

        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(files)
        split = self._get_splits(len(files))
        if self.train:
            return files[split:]
        return files[:split]
    
    @staticmethod
    def _get_splits(total_examples, val_ratio=0.1):
        assert ((val_ratio >= 0) and (val_ratio <= 1)), "[!] valid_size should be in the range [0, 1]."
        return int(np.floor(val_ratio * total_examples))


class BagXRay_dataset(Dataset):
    """
    Class for loading Xray views from ct baggage
    """
    def __init__(self,
                 data_dir,
                 train,
                 split_shuffle=True,
                 split_seed=500,
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

        :return data: (dict)
        'xrays': torch.Tensor(channels, scale, scale)
        """

        self.data_dir = data_dir
        self.train = train
        self.shuffle = split_shuffle
        self.seed = split_seed
        self.types = types
        self.direction = direction
        self.degrees = degrees

        self.split_dirs = self._get_dirs()
        self.xray_imgs = self._get_files()
        
        self.xray_tx = [transforms.Resize(scale),
                        Normalization(0, 255),
                        ToTensor()]

    def __len__(self):
        return len(self.xray_imgs)

    def __getitem__(self, idx):
        xray_file = self.xray_imgs[idx]
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
        data_dirs = sorted(os.listdir(self.data_dir))
        dirs = [os.path.join(self.data_dir, x) for x in data_dirs if "projections_grayscale2" in set(os.listdir(os.path.join(self.data_dir, x)))]
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(dirs)
        split = self._get_splits(len(dirs))
        if self.train:
            return dirs[split:]
        return dirs[:split]
        
    @staticmethod
    def _get_splits(total_examples, val_ratio=0.1):
        assert ((val_ratio >= 0) and (val_ratio <= 1)), "[!] valid_size should be in the range [0, 1]."
        return int(np.floor(val_ratio * total_examples))

    def _get_files(self):
        files = [os.path.join(d, f"projections_{ch}2/{os.path.basename(d)}_{cam}_{90}.png") 
                 for d in self.split_dirs
                 for ch in (['rgb', 'grayscale'] if self.types == 'both' else [self.types])
                 for cam in (['azimuth', 'elevation'] if self.direction == 'both' else [self.direction])]
        return files


class BagCT_dataset(Dataset):
    """
    Class for loading CT scans
    for baggage security dataset
    """
    def __init__(self, data_dir, train, split_shuffle=True, split_seed=500, scale=128,
                scale_ct=True, use_f16=True):
    
        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.train = train
        self.seed = split_seed
        self.shuffle = split_shuffle
        self.data_dir = data_dir
        self.f16 = use_f16

        self.split_dirs = self._get_dirs()

        self.ct_tx = [Resize_image((scale, scale, scale), f16=self.f16) if scale_ct else ReturnIdentity(),
                      Normalization_min_max(0., 1.),
                      ToTensor(f16=self.f16)]

        
    def __len__(self):
        print("|_Total files: ", len(self.split_dirs))
        return len(self.split_dirs)

    def __getitem__(self, idx):
        object_dir = os.path.dirname(self.split_dirs[idx])
        ct_file = self.split_dirs[idx]
        
        ct_scan = np.load(ct_file)['ct'] #if self.f16 else np.load(ct_file)

        for transf in self.ct_tx:
            ct_scan = transf(ct_scan)

        ct_scan = ct_scan.unsqueeze(0)
        
        data = {
            "ct": ct_scan,
            "dir_name": object_dir}

        return data
        
    def _get_dirs(self):
        data_dirs = [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        data_dirs.sort()
        files = []

        pattern = '*_f16.npz' #if self.f16 else '*.npy'

        for folder in data_dirs:
            file_list = list(Path(folder).rglob(pattern))
            if len(file_list) > 0 and "projections_grayscale2" in set(os.listdir(folder)):
                files.append(file_list[0])

        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(files)
        split = self._get_splits(len(files))
        if self.train:
            return files[split:]
        return files[:split]
    
    @staticmethod
    def _get_splits(total_examples, val_ratio=0.1):
        assert ((val_ratio >= 0) and (val_ratio <= 1)), "[!] valid_size should be in the range [0, 1]."
        return int(np.floor(val_ratio * total_examples))


class CT_dataset(Dataset):
    """
    Class for loading CT scans
    """
    def __init__(self, data_dir, train, load_res=None, scale=256,
                 scale_ct=False, dataset='chest',
                 use_f16=True):
        """
        :param data_dir: (str) path to folder containing all ct scans
        :param train: (bool) are we training or testing.
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
        split_file = f'data/{self.dataset}_train_split.txt' if train else f'data/{self.dataset}_test_split.txt'
        self.data_dir = data_dir
        self.load_res = load_res

        self.split_list = self._get_split(split_file)
        self.object_dirs = self._get_dirs()
        self.f16 = use_f16

        self.ct_tx = [Resize_image((scale, scale, scale), f16=self.f16) if scale_ct else nn.Identity(),
                      Normalization_min_max(0., 1., remove_noise=False), 
                      ToTensor(f16=self.f16)]


    def __len__(self):
        return len(self.object_dirs)

    def __getitem__(self, idx):
        object_dir = self.object_dirs[idx]

        if self.load_res is not None:
            ct_path = f'{object_dir}/*_{self.load_res}.npz'
            ct_file = glob.glob(ct_path)
            ct_scan = np.load(ct_file)['ct'] if self.f16 else np.load(ct_file)
        else:
            pattern = f'*{self.dataset}_f16.npz' if self.f16 else f'*{self.dataset}128_f16.npz'
            ct_path = f'{object_dir}/{pattern}'
            ct_file = glob.glob(ct_path)
            ct_scan = np.flip(np.load(ct_file[0])['ct'])

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


class Dataset2D(Dataset):
    """
    Class for loading views from SHREC16
    """
    def __init__(self,
                 data_dir,
                 train,
                 split_shuffle=True,
                 split_seed=500,
                 scale=128,
                 projections=2):
        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.projections = projections
        self.data_dir = data_dir

        self.train = train
        self.shuffle = split_shuffle
        self.seed = split_seed
        
        self.split_dirs = self._get_dirs()
        self.imgs = self._get_files()
        
        self.tx = [transforms.Resize(scale) if scale != 128 else ReturnIdentity(),
                   Normalization(0, 255),
                   ToTensor()]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_file = self.imgs[idx]
        img = Image.open(img_file).convert('L')

        for transf in self.tx:
            img = transf(img)

        data = {
            "xray": gaussian_filter(img.unsqueeze(0), sigma=2)
        }

        return data

    def _get_dirs(self):
        data_dirs = sorted(os.listdir(self.data_dir))
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(data_dirs)
        split = self._get_splits(len(data_dirs))
        if self.train:
            return data_dirs[split:]
        return data_dirs[:split]
        
    @staticmethod
    def _get_splits(total_examples, val_ratio=0.1):
        assert ((val_ratio >= 0) and (val_ratio <= 1)), "[!] valid_size should be in the range [0, 1]."
        return int(np.floor(val_ratio * total_examples))

    def _get_files(self):
        files = []

        for object_dir in self.split_dirs:
            files = files + sorted(glob.glob(os.path.join(self.data_dir, object_dir, '*.png')))

        return files


class Dataset3D(Dataset):
    """
    Class for loading CT scans
    """
    def __init__(self, data_dir, train, split_shuffle=True, split_seed=500, load_res=None, scale=128, scale_ct=False, use_f16=True):
        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.train = train
        self.seed = split_seed
        self.shuffle = split_shuffle
        self.data_dir = data_dir
        self.load_res = load_res
        self.f16 = use_f16

        self.split_dirs = self._get_dirs()

        self.ct_tx = [Resize_image((scale, scale, scale), f16=self.f16) if scale_ct else ReturnIdentity(),
                      Normalization_min_max(0., 1., remove_noise=False),
                      ToTensor(f16=self.f16)]

        
    def __len__(self):
        print("|_Total files: ", len(self.split_dirs))
        return len(self.split_dirs)

    def __getitem__(self, idx):
        object_dir = os.path.dirname(self.split_dirs[idx][0])
        ct_file = self.split_dirs[idx][0]
        
        ct_scan = np.load(ct_file)['ct']

        for transf in self.ct_tx:
            ct_scan = transf(ct_scan)

        ct_scan = gaussian_filter(ct_scan.unsqueeze(0), sigma=2)
              
        data = {
            "ct": ct_scan,
            "dir_name": object_dir}

        return data
        
    def _get_dirs(self):
        data_dirs = [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        data_dirs.sort()
        files = []

        for folder in data_dirs:
            file_list = glob.glob(f'{folder}/*.npz')
            if len(file_list) > 0:
                files.append(file_list)

        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(files)
        split = self._get_splits(len(files))
        if self.train:
            return files[split:]
        return files[:split]
    
    @staticmethod
    def _get_splits(total_examples, val_ratio=0.1):
        assert ((val_ratio >= 0) and (val_ratio <= 1)), "[!] valid_size should be in the range [0, 1]."
        return int(np.floor(val_ratio * total_examples))

    
class SHREC16_Dataset(Dataset):
    """
    Class for loading data from SHREC16_Dataset
    """
    def __init__(self, data_dir, train, split_shuffle=True, split_seed=500, load_res=None, scale=128, scale_ct=False, use_f16=False):
    
        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.train = train
        self.seed = split_seed
        self.shuffle = split_shuffle
        self.data_dir = data_dir
        self.load_res = load_res
        self.f16 = use_f16

        self.split_dirs = self._get_dirs()

        self.ct_tx = [Resize_image((scale, scale, scale)) if scale_ct else ReturnIdentity(),
                      Normalization_min_max(0., 1., remove_noise=False),
                      ToTensor(f16=self.f16)]

        self.xray_tx = torchvision.transforms.Compose([
            torchvision.transforms.Resize(scale),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.],
                                             std=[1.])
        ])

        
    def __len__(self):
        print("|_Total files: ", len(self.split_dirs))
        return len(self.split_dirs)

    def __getitem__(self, idx):
        object_dir = os.path.dirname(self.split_dirs[idx])
        ct_file = self.split_dirs[idx]
        ct_scan = np.load(ct_file)['ct']

        for transf in self.ct_tx:
            ct_scan = transf(ct_scan)

        xray_files = sorted(glob.glob(os.path.join(object_dir, '*.png')))
        xray_list = []
        for xray in xray_files:
            img = Image.open(xray).convert('L')
            img = self.xray_tx(img)
            xray_list.append(img)
        xrays = torch.stack(xray_list, 0)

        ct_scan = gaussian_filter(ct_scan.unsqueeze(0), sigma=1)
              
        data = {
            "xrays": xrays,
            "ct": ct_scan,
            "dir_name": object_dir}

        return data
        
    def _get_dirs(self):
        data_dirs = [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        data_dirs.sort()
        files = []

        pattern = '*.npz'

        for folder in data_dirs:
            file_list = list(Path(folder).rglob(pattern))
            if len(file_list) > 0:
                files.append(file_list[0])

        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(files)
        split = self._get_splits(len(files))
        if self.train:
            return files[split:]
        return files[:split]
    
    @staticmethod
    def _get_splits(total_examples, val_ratio=0.1):
        assert ((val_ratio >= 0) and (val_ratio <= 1)), "[!] valid_size should be in the range [0, 1]."
        return int(np.floor(val_ratio * total_examples))
