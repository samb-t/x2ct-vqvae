import torch
import math
from torch import nn
from torch.nn.utils import spectral_norm


class Discriminator3D(nn.Module):
    def __init__(self, ndf=64, nc=1, im_size=128):
        super().__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        #nfc={4: 1024, 8: 1024, 16: 512, 32: 256, 64: 128, 128: 64, 256: 32, 512: 16, 1024: 8}

        self.down_from_big = nn.Sequential( 
                Conv(True, nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True) )

        self.down_4 = DownBlockComp(nfc[512], nfc[256], True)
        self.down_8 = DownBlockComp(nfc[256], nfc[128], True)

        self.rf_big = nn.Sequential(
                            Conv(True, nfc[128], nfc[32], 1, 1, 0, bias=False),
                            nn.LeakyReLU(0.2, inplace=True),
                            Conv(True, nfc[32], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[256], use3d=True)
        self.se_4_32 = SEBlock(nfc[256], nfc[128], use3d=True)
        
        
    def forward(self, imgs, is_fake=None, part=None):
        feat_2 = self.down_from_big(imgs)
        feat_4 = self.down_4(feat_2)
        feat_16 = self.se_2_16(feat_2, feat_4)
        
        feat_8 = self.down_8(feat_16)
        feat_last = self.se_4_32(feat_4, feat_8)
        
        rf_0 = self.rf_big(feat_last).view(-1)

        return rf_0


class Discriminator2D(nn.Module):
    def __init__(self, ndf=64, nc=1, im_size=128):
        super().__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        #nfc={4: 1024, 8: 1024, 16: 512, 32: 256, 64: 128, 128: 64, 256: 32, 512: 16, 1024: 8}

        self.down_from_big = nn.Sequential( 
                Conv(False, nc, nfc[512], 3, 1, 1),
                nn.LeakyReLU(0.2) )

        self.down_4 = DownBlockComp(nfc[512], nfc[256])
        self.down_8 = DownBlockComp(nfc[256], nfc[128])

        self.rf_big = nn.Sequential(
                            Conv(False, nfc[128], nfc[32], 1, 1, 0, bias=False),
                            nn.LeakyReLU(0.2),
                            Conv(False, nfc[32], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[256])
        self.se_4_32 = SEBlock(nfc[256], nfc[128])
        
        self.decoder_big = DynamicSimpleDecoder(nfc[128], nc, out_size=self.im_size)
        self.decoder_part = DynamicSimpleDecoder(nfc[256], nc,
                                          out_size=self.im_size)
        
    def forward(self, imgs, is_fake, part=None):
        feat_2 = self.down_from_big(imgs)
        feat_4 = self.down_4(feat_2)
        feat_16 = self.se_2_16(feat_2, feat_4)
        
        feat_8 = self.down_8(feat_16)
        feat_last = self.se_4_32(feat_4, feat_8)
        
        rf_0 = self.rf_big(feat_last).view(-1)

        if not is_fake:
            rec_img_big = self.decoder_big(feat_last)

            assert part is not None
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(feat_16[:,:,:8,:8])
            if part==1:
                rec_img_part = self.decoder_part(feat_16[:,:,:8,8:])
            if part==2:
                rec_img_part = self.decoder_part(feat_16[:,:,8:,:8])
            if part==3:
                rec_img_part = self.decoder_part(feat_16[:,:,8:,8:])

            return rf_0, [rec_img_big, rec_img_part]

        return rf_0


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes, use3d=False):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            Conv(use3d, in_planes, out_planes, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            Conv(use3d, out_planes, out_planes, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            AvgPool(use3d, 2, 2),
            Conv(use3d, in_planes, out_planes, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out, use3d=False):
        super().__init__()

        self.main = nn.Sequential(AdaptiveAvgPool(use3d, 4), 
                                  Conv(use3d, ch_in, ch_out, 4, 1, 0, bias=False),
                                  Swish(),
                                  Conv(use3d, ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

        #feat_small -> x_low   feat_big -> x_high
    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)#This product is performed channel-wise


def AvgPool(use3d, *args, **kwargs):
    if use3d:
        return nn.AvgPool3d(*args, **kwargs)
    return nn.AvgPool2d(*args, **kwargs)


def AdaptiveAvgPool(use3d, *args, **kwargs):
    if use3d:
        return nn.AdaptiveAvgPool3d(*args, **kwargs)
    return nn.AdaptiveAvgPool2d(*args, **kwargs)


def Conv(use3d, *args, **kwargs):
    if use3d:
        return spectral_norm(nn.Conv3d(*args, **kwargs))
    return spectral_norm(nn.Conv2d(*args, **kwargs))

class DynamicSimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3, out_size=64, use3d=False):
        super().__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                Upsample(use3d=use3d),
                Conv(use3d, in_planes, out_planes*2, 3, 1, 1, bias=False),
                BatchNorm(use3d, out_planes*2), nn.GLU(dim=1))
            return block

        self.layer1 = nn.Sequential(AdaptiveAvgPool(use3d, 8),
                                    upBlock(nfc_in, nfc[16]))
        self.upblocks = []
        for i in range(4, int(math.log(out_size,2))):
            self.upblocks.append(upBlock(nfc[2**i], nfc[2**(i+1)]))
            
        self.main = nn.Sequential(self.layer1,
                                  *self.upblocks,
                                  Conv(use3d, nfc[out_size], nc, 3, 1, 1, bias=False))

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)


def Upsample(use3d=False):
    if use3d:
        return nn.Upsample(scale_factor=2, mode='trilinear')
    return nn.Upsample(scale_factor=2, mode='nearest')


def BatchNorm(use3d, *args, **kwargs):
    if use3d:
        return nn.BatchNorm3d(*args, **kwargs)
    return nn.BatchNorm2d(*args, **kwargs)


