import torch
import torch.nn as nn
import numpy as np
from kornia.augmentation import (
    RandomDepthicalFlip3D,
    RandomHorizontalFlip3D,
    RandomVerticalFlip3D,
    RandomRotation3D,
    RandomAffine3D,
    RandomPerspective,
    RandomAffine,
    RandomRotation,
    RandomPlasmaContrast,
)
import math

class AdaptiveDiscriminatorAugmentation(nn.Module):
    """
    This class implements adaptive discriminator augmentation proposed in:
    StyleGAN2-ADA https://arxiv.org/pdf/2006.06676.pdf
    The adaptive discriminator augmentation model wraps a given discriminator network.
    ref: https://github.com/ChristophReich1996/Multi-StyleGAN
    kornia 0.6.6 used
    """
    def __init__(self, discriminator, use_3d=True, progressive_d=True,
                 r_target=0.6, p_step=5e-03, r_update=8, p_max=0.8):
        """
        Constructor method
        :param discriminator: (Union[nn.Module, nn.DataParallel]) Discriminator network
        :param r_target: (float) Target value for r
        :param p_step: (float) Step size of p
        :param r_update: (int) Update frequency of r
        :param p_max: (float) Global max value of p
        """
        # Call super constructor
        super().__init__()
        # Save parameters
        self.discriminator = discriminator
        self.is_3d = use_3d
        self.progressive = progressive_d
        self.r_target = r_target
        self.p_step = p_step
        self.r_update = r_update
        self.p_max = p_max
        # Init augmentation variables
        self.r = []
        self.p = 0.05
        self.r_history = []
        # Init augmentation pipeline
        self.augmentation_pipeline = AugmentationPipeline3D() if use_3d else AugmentationPipeline2D()
        

    @torch.no_grad()
    def __calc_r(self, prediction):
        """
        Method computes the overfitting heuristic r.
        :param prediction: (torch.Tensor) Scalar prediction [batch size, 1]
        :return: (float) Value of the overfitting heuristic r
        """
        self.r.append(torch.mean(torch.sign(prediction)).item())


    def forward(self, images, is_fake=False, part=None):
        """
        Forward pass
        :param images: (torch.Tensor) Mini batch of images (real or fake) [batch size, channels, height, width]
        :return: real/fake prediction of the discriminator
        """
        # Apply augmentations
        images = self.augmentation_pipeline(images, self.p)
        pred = self.discriminator(images, is_fake, part) if self.progressive else self.discriminator(images)
        if is_fake:
            self.__calc_r(pred.detach()) if not self.progressive or self.is_3d else self.__calc_r(pred[0].detach())

        # Update p
        if len(self.r) >= self.r_update:
            # Calc r over the last epochs
            r = np.mean(self.r)
            # If r above target value increment p else reduce
            if r > self.r_target:
                self.p += self.p_step
            else:
                self.p -= self.p_step
            # Check if p is negative
            self.p = self.p if self.p >= 0. else 0.
            # Check if p is larger than 1
            self.p = self.p if self.p < self.p_max else self.p_max
            # Reset r
            self.r = []
            # Save current r in history
            self.r_history.append(r)

        if self.progressive and not is_fake and not self.is_3d:
            return pred[0], pred[-1]
        return pred


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
        


class AugmentationPipeline2D(nn.Module):
    """
    This class implement the differentiable augmentation pipeline for ADA (2D transforms).
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
        :param images: (torch.Tensor) Mini-batch images [batch size, channels, height, width]
        :param p: (float) Probability of augmentation to be applied
        :return: (torch.Tensor) Augmented images [batch size, channels, height, width]
        """
        # Appply perspective
        images_persp = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_persp) > 0:
            perspective = RandomPerspective(p=1., distortion_scale=0.2, keepdim=True)
            images[images_persp] = self.apply_aug(images[images_persp], perspective)
        # Perform translations
        images_translated = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_translated) > 0:
            translate = RandomAffine(p=1., degrees=0, translate=(0.125, 0.125), keepdim=True)
            images[images_translated] = self.apply_aug(images[images_translated], translate)

        # Perform scale
        images_scaled = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_scaled) > 0:
            scale = np.random.lognormal(mean=0, sigma=(0.2 * math.log(2)) ** 2)
            scaling = (1.0, scale) if scale > 1.0 else (scale, 1.0)
            scale = RandomAffine(p=1., degrees=0, scale=scaling, keepdim=True)
            images[images_scaled] = self.apply_aug(images[images_scaled], scale)

        # Apply rotation
        images_rotated = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_rotated) > 0:
            # 3D random rotation between -15, 15 degrees for yaw, pitch, roll
            randrot = RandomRotation(15, p=1.0)
            images[images_rotated] = self.apply_aug(images[images_rotated], randrot)
            

        # Apply plasma random Contrast
        images_contrasted = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_contrasted) > 0:
            plasma = RandomPlasmaContrast(p=1.0)
            images[images_contrasted] = self.apply_aug(images[images_contrasted], plasma)

        return images


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3, 4], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x
