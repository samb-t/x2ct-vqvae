import numpy as np
from skimage.metrics import structural_similarity as calc_ssim


def MAE(arr1, arr2, size_average=True):
    '''
    :param arr1:
      Format-[NDHW], OriImage
    :param arr2:
      Format-[NDHW], ComparedImage
    :return:
      Format-None if size_average else [N]
    '''
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    if size_average:
        return np.abs(arr1 - arr2).mean()
    else:
        return np.abs(arr1 - arr2).mean(1).mean(1).mean(1)


def MSE(arr1, arr2, size_average=True):
    '''
    :param arr1:
    Format-[NDHW], OriImage
    :param arr2:
    Format-[NDHW], ComparedImage
    :return:
    Format-None if size_average else [N]
    '''
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    if size_average:
        return np.power(arr1 - arr2, 2).mean()
    else:
        return np.power(arr1 - arr2, 2).mean(1).mean(1).mean(1)


def Structural_Similarity(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    '''
    :param arr1:
    Format-[NDHW], OriImage [0,1]
    :param arr2:
    Format-[NDHW], ComparedImage [0,1]
    :return:
    Format-None if size_average else [N]
    '''
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    N = arr1.shape[0]
    # Depth
    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = []
    for i in range(N):
        ssim = calc_ssim(arr1_d[i], arr2_d[i], data_range=PIXEL_MAX, channel_axis=1)
        ssim_d.append(ssim)
    ssim_d = np.asarray(ssim_d, dtype=np.float64)

    # Height
    arr1_h = np.transpose(arr1, (0, 1, 3, 2))
    arr2_h = np.transpose(arr2, (0, 1, 3, 2))
    ssim_h = []
    for i in range(N):
        ssim = calc_ssim(arr1_h[i], arr2_h[i], data_range=PIXEL_MAX, channel_axis=1)
        ssim_h.append(ssim)
    ssim_h = np.asarray(ssim_h, dtype=np.float64)

    # Width
    # arr1_w = np.transpose(arr1, (0, 1, 2, 3))
    # arr2_w = np.transpose(arr2, (0, 1, 2, 3))
    ssim_w = []
    for i in range(N):
        ssim = calc_ssim(arr1[i], arr2[i], data_range=PIXEL_MAX, channel_axis=1)
        ssim_w.append(ssim)
    ssim_w = np.asarray(ssim_w, dtype=np.float64)

    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

    if size_average:
        return [ssim_d.mean(), ssim_h.mean(), ssim_w.mean(), ssim_avg.mean()]
    else:
        return [ssim_d, ssim_h, ssim_w, ssim_avg]



def Peak_Signal_to_Noise_Rate_3D(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
  '''
  :param arr1:
    Format-[NDHW], OriImage [0,1]
  :param arr2:
    Format-[NDHW], ComparedImage [0,1]
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)
  eps = 1e-10
  se = np.power(arr1 - arr2, 2)
  mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
  zero_mse = np.where(mse == 0)
  mse[zero_mse] = eps
  psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
  # #zero mse, return 100
  psnr[zero_mse] = 100

  if size_average:
    return psnr.mean()
  else:
    return psnr


def Peak_Signal_to_Noise_Rate(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    '''
    :param arr1:
    Format-[NDHW], OriImage [0,1]
    :param arr2:
    Format-[NDHW], ComparedImage [0,1]
    :return:
    Format-None if size_average else [N]
    '''
    assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    # Depth
    mse_d = se.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True).squeeze(3).squeeze(2)
    zero_mse = np.where(mse_d==0)
    mse_d[zero_mse] = eps
    psnr_d = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_d))
    # #zero mse, return 100
    psnr_d[zero_mse] = 100
    psnr_d = psnr_d.mean(1)

    # Height
    mse_h = se.mean(axis=1, keepdims=True).mean(axis=3, keepdims=True).squeeze(3).squeeze(1)
    zero_mse = np.where(mse_h == 0)
    mse_h[zero_mse] = eps
    psnr_h = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_h))
    # #zero mse, return 100
    psnr_h[zero_mse] = 100
    psnr_h = psnr_h.mean(1)

    # Width
    mse_w = se.mean(axis=1, keepdims=True).mean(axis=2, keepdims=True).squeeze(2).squeeze(1)
    zero_mse = np.where(mse_w == 0)
    mse_w[zero_mse] = eps
    psnr_w = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_w))
    # #zero mse, return 100
    psnr_w[zero_mse] = 100
    psnr_w = psnr_w.mean(1)

    psnr_avg = (psnr_h + psnr_d + psnr_w) / 3
    if size_average:
        return [psnr_d.mean(), psnr_h.mean(), psnr_w.mean(), psnr_avg.mean()]
    return [psnr_d, psnr_h, psnr_w, psnr_avg]
