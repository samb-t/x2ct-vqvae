import torch
import torch.nn as nn
import numpy as np
from kornia.augmentation import (
    RandomDepthicalFlip3D,
    RandomHorizontalFlip3D,
    RandomVerticalFlip3D,
    RandomRotation3D,
    RandomAffine3D,
)
import math


class AdaptiveDiscriminatorAugmentation(nn.Module):
    """
    This class implements adaptive discriminator augmentation proposed in:
    StyleGAN2-ADA https://arxiv.org/pdf/2006.06676.pdf
    The adaptive discriminator augmentation model wraps a given discriminator network.
    ref: https://github.com/ChristophReich1996/Multi-StyleGAN
    adapted for CT suitable transforms
    kornia 0.6.6 used
    """
    def __init__(self, discriminator, r_target=0.6, p_step=5e-03, r_update=8, p_max=0.6):
        """
        Constructor method
        :param discriminator: (Union[nn.Module, nn.DataParallel]) Discriminator network
        :param r_target: (float) Target value for r
        :param p_step: (float) Step size of p
        :param r_update: (int) Update frequency of r
        :param p_max: (float) Global max value of p (augmentation strength)
        """
        # Call super constructor
        super().__init__()
        # Save parameters
        self.discriminator = discriminator
        self.r_target = r_target
        self.p_step = p_step
        self.r_update = r_update
        self.p_max = p_max
        # Init augmentation variables
        self.r = []
        self.p = 0.05
        self.r_history = []
        # Init augmentation pipeline
        self.augmentation_pipeline = AugmentationPipeline3D()
        

    @torch.no_grad()
    def __calc_r(self, prediction):
        """
        Method computes the overfitting heuristic r.
        :param prediction: (torch.Tensor) Scalar prediction [batch size, 1]
        :return: (float) Value of the overfitting heuristic r
        """
        self.r.append(torch.mean(torch.sign(prediction)).item())


    def update_p(self):
        # Calculate r over the last epochs
        r = np.mean(self.r)
        # If r above target value increment p else reduce
        self.p += self.p_step if r > self.r_target else -self.p_step
        # Clip p between 0 and p_max
        self.p = min(max(self.p, 0), self.p_max)
        # Reset r
        self.r = []
        # Save current r in history
        self.r_history.append(r)


    def forward(self, images, is_fake=False):
        """
        Forward pass
        :param images: (torch.Tensor) Mini batch of images (real or fake) [batch size, channels, height, width]
        :return: real/fake prediction of the discriminator
        """
        # Apply augmentations
        images = self.augmentation_pipeline(images, self.p)

        pred, activations = self.discriminator(images)

        if is_fake:
            self.__calc_r(pred.detach())

        # Update p
        if len(self.r) >= self.r_update:
            self.update_p()

        return pred, activations


class AugmentationPipeline3D(nn.Module):
    """
    This class implement the differentiable augmentation pipeline for ADA.
    """

    def __init__(self):
        # Call super constructor
        super().__init__()

    def apply_aug(self, x, transform):
        if x.dtype == torch.float16:
            return transform(x).half()
        return transform(x)

    def forward(self, images, p):
        """
        Forward pass applies augmentation to mini-batch of given images
        :param images: (torch.Tensor) Mini-batch images [batch size, channels, depth, height, width]
        :param p: (float) Probability of augmentation to be applied
        :return: (torch.Tensor) Augmented images [batch size, channels, depth, height, width]
        """
        # Perform depthical flip
        images_flipped = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_flipped) > 0:
            dflip = RandomDepthicalFlip3D(p=1.)
            images[images_flipped] = self.apply_aug(images[images_flipped], dflip)

        # Perform rotation
        images_rotated = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_rotated) > 0:
            # 3D random rotation between -15, 15 degrees for yaw, pitch, roll
            randrot = RandomRotation3D(15, p=1.0)
            images[images_rotated] = self.apply_aug(images[images_rotated], randrot)

        # Perform vertical flip
        images_translated = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_translated) > 0:
            vflip = RandomVerticalFlip3D(p=1.)
            images[images_translated] = self.apply_aug(images[images_translated], vflip)
        # Perform horizontal flip
        images_htranslated = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_htranslated) > 0:
            hflip = RandomHorizontalFlip3D(p=1.)
            images[images_htranslated] = self.apply_aug(images[images_htranslated], hflip)

        # Perform isotropic scaling
        images_scaling = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_scaling) > 0:
            scale = np.random.lognormal(mean=0, sigma=(0.2 * math.log(2)) ** 2)
            scaling = (1.0, scale) if scale > 1.0 else (scale, 1.0)
            isoscale = RandomAffine3D(degrees=0., scale=scaling, p=1.)
            images[images_scaling] = self.apply_aug(images[images_scaling], isoscale)

        # Perform anisotropic scaling
        images_scaling = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_scaling) > 0:
            scale_a = np.random.lognormal(mean=0, sigma=(0.2 * math.log(2)) ** 2)
            scale_b = np.random.lognormal(mean=0, sigma=(0.2 * math.log(2)) ** 2)
            scale_c = np.random.lognormal(mean=0, sigma=(0.2 * math.log(2)) ** 2)
            anscale = RandomAffine3D(degrees=0., scale=((1., scale_a), (1., scale_b), (1., scale_c)), p=1.)
            images[images_scaling] = self.apply_aug(images[images_scaling], anscale)
        
        # Apply random Contrast
        images_contrasted = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_contrasted) > 0:
            images[images_contrasted] = rand_contrast(images[images_contrasted])

        return images


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3, 4], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x
