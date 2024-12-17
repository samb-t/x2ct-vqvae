import copy
import torch
import torch_fidelity
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from .log_utils import load_model, load_stats, log, save_images


def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    Modified according to detectron2's fvcore,
    refer to https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if size_average:
        return loss.mean()
    return loss.sum()


class Mish(nn.Module):
    def forward(self, x):
        return F.mish(x)

class AdaptivePseudoAugment:
    def __init__(self, start_epoch=14, initial_prob=0., threshold=0.4, iterations=4, max_prob=0.6):
        self.start = start_epoch
        self.init_prob = initial_prob
        self.t = threshold # lower value is suggested for smaller datasets
        self.speed = 1e-6
        self.it = iterations
        self.grad_accumulation = 5
        self.max_prob = max_prob
        self.lambda_rs = []
        self.lambda_fs = []


    def update_lambdas(self, batch_size, num_mix_fakes, logits_real, logits_fake):
        _lambda_r = logit_sigmoid(logits_real[:batch_size - num_mix_fakes]).sign().mean().item()
        self.lambda_rs.append(_lambda_r)
        _lambda_f = logit_sigmoid(logits_fake[:batch_size]).sign().mean().item()
        self.lambda_fs.append(_lambda_f)
        
    def adjust_prob(self, batch_size):
        if len(self.lambda_rs) != 0 or len(self.lambda_fs) != 0:
            lambda_r = sum(self.lambda_rs) / len(self.lambda_rs)
            lambda_f = sum(self.lambda_fs) / len(self.lambda_fs)
            lambda_rf = (lambda_r - lambda_f) / 2 # this can be used instead of λr for adjusting
            self.init_prob += \
                np.sign(lambda_rf - self.t) * \
                self.speed * \
                batch_size * self.grad_accumulation * self.it
            self.init_prob = np.clip(self.init_prob, 0., self.max_prob)
            self.lambda_rs = []


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, d, h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*d*h*w)


class DensityLoss(nn.Module):
    def __init__(self, threshold, eps=1e-6):
        super(DensityLoss, self).__init__()
        self.threshold = threshold
        self.eps = eps

    def update_threshold(self, threshold_delta):
        if self.threshold > 0.0:
            self.threshold -= threshold_delta

    def forward(self, x, y):
        if self.threshold > 0.0:
            mask = nn.Threshold(self.threshold, self.eps)
            x = mask(x)
        b, c, d, h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*d*h*w)


def logit_sigmoid(x):
    return -F.softplus(x) + F.softplus(x)
    
def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32 if 32 < in_channels else 1, num_channels=in_channels, eps=1e-6, affine=True)


@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


#@torch.jit.script
def hinge_d_loss(logits_real, logits_fake, return_losses=False):
    loss_real = torch.mean(F.mish(1. - logits_real))
    loss_fake = torch.mean(F.mish(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    if return_losses:
        return d_loss, loss_real, loss_fake
    return d_loss


def calculate_adaptive_weight(recon_loss, g_loss, last_layer, disc_weight_max):
    recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
    return d_weight


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)


# TODO: replace this with general checkpointing method
def load_vqgan_from_checkpoint(H, vqgan, optim, disc_optim, ema_vqgan):
    if vqgan is not None:
        vqgan = load_model(vqgan, "vqgan", H.train.load_step, f"{H.run.name}_{H.run.experiment}").cuda()
    if optim is not None:
        optim = load_model(optim, "ae_optim", H.train.load_step, f"{H.run.name}_{H.run.experiment}")
    if disc_optim is not None:
        disc_optim = load_model(disc_optim, "disc_optim", H.train.load_step, f"{H.run.name}_{H.run.experiment}")

    if ema_vqgan is not None:
        try:
            ema_vqgan = load_model(ema_vqgan, "vqgan_ema", H.train.load_step, f"{H.run.name}_{H.run.experiment}")
        except FileNotFoundError:
            log("No EMA model found, starting EMA from model load point", level="warning")
            ema_vqgan = copy.deepcopy(vqgan)

    return vqgan, optim, disc_optim, ema_vqgan


# def calc_FID(H, model):
#     # generate_recons(H, model)
#     real_dataset, _ = get_datasets(H.dataset, H.img_size, custom_dataset_path=H.custom_dataset_path)
#     real_dataset = NoClassDataset(real_dataset)
#     recons = BigDataset(f"logs/{H.log_dir}/FID_recons/images/")
#     fid = torch_fidelity.calculate_metrics(
#         input1=recons,
#         input2=real_dataset,
#         cuda=True,
#         fid=True,
#         verbose=True,
#         input2_cache_name=f"{H.dataset}_cache" if H.dataset != "custom" else None,
#     )["frechet_inception_distance"]

#     return fid


# @torch.no_grad()
# def generate_recons(H, model):
#     # if using validation on FFHQ, don't want to include validation set images in FID calc
#     training_with_validation = True if H.steps_per_eval else False

#     data_loader, _ = get_data_loaders(
#         H.dataset,
#         H.img_size,
#         H.batch_size,
#         get_val_dataloader=training_with_validation,
#         drop_last=False,
#         shuffle=False,
#     )
#     log("Generating recons for FID calculation")

#     for idx, x in tqdm(enumerate(iter(data_loader))):
#         x = x[0].cuda()  # TODO check this for multiple datasets
#         x_hat, *_ = model.ae(x)
#         save_images(x_hat, "recon", idx, f"{H.log_dir}/FID_recons", save_individually=True)
