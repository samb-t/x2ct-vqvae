import numpy as np
import SimpleITK as sitk


def to_unNorm(ct, pred_ct, inverse=False):
    fake_ct = pred_ct.clone().detach().cpu().numpy()
    real_ct = ct.clone().detach().cpu().numpy()

    fake_ct_t = np.transpose(fake_ct, (0, 2, 1, 3))
    real_ct_t = np.transpose(real_ct, (0, 2, 1, 3))

    if inverse:
        fake_ct_t = fake_ct_t[:, ::-1, :, :]
        real_ct_t = real_ct_t[:, ::-1, :, :]

    fake_ct_t = toUnnormalize(fake_ct_t, 0., 1.)
    real_ct_t = toUnnormalize(real_ct_t, 0., 1.)

    fake_ct_t = np.clip(fake_ct_t, 0, 1)

    return real_ct_t, fake_ct_t


def toUnnormalize(img, mean, std):
    img = img * std + mean
    return img


def back_to_HU(input_image, min=0, max=2000):
    image = input_image * (max - min) + min
    return image


def save_volume(nparray, name, mha=True):
    np.save(name + '.npy', nparray)
    nparray.squeeze(0).astype('int16').tofile(name + '.raw')
    if mha:
        save_mha(nparray.squeeze(0), spacing=(1., 1., 1.), origin=(0, 0, 0), path=name+'.mha')


# For visualising using e.g 3DSlicer
def save_mha(volume, spacing, origin, path):
    itkimage = sitk.GetImageFromArray(volume, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, path, True)
