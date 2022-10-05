'''
VQGAN code, originally from Taming Transformers, adapted for Unleashing Transformers
'''

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffaug import DiffAugment
from utils.vqgan_utils import normalize, swish, adopt_weight, hinge_d_loss, calculate_adaptive_weight
from utils.log_utils import log

class VQGAN(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.ae = VQAutoEncoder(H)
        self.disc = Discriminator(
            H.data.channels,
            H.model.ndf,
            n_layers=H.model.disc_layers
        )
        self.perceptual = None
        if H.model.perceptual_loss:
            self.perceptual = lpips.LPIPS(net="vgg")
            self.perceptual_weight = H.model.perceptual_weight
        self.disc_start_step = H.model.disc_start_step
        self.disc_weight_max = H.model.disc_weight_max
        self.diffaug_policy = H.model.diffaug_policy

    def train_iter_together(self, x, step):
        """
        This is the training iteration we used in Unleashing Transformers, however,
        wondering if we might get better results if we do what most papers do
        and actually train the GAN iteritvely rathen than effectively with one optimiser 
        in conjunction with .detach()
        """
        stats = {}
        # update gumbel softmax temperature based on step. Anneal from 1 to 1/16 over 150000 steps
        if self.ae.quantizer_type == "gumbel":
            self.ae.quantize.temperature = max(1/16, ((-1/160000) * step) + 1)
            stats["gumbel_temp"] = self.ae.quantize.temperature

        x_hat, codebook_loss, quant_stats = self.ae(x)

        # get recon/perceptual loss
        recon_loss = torch.abs(x.contiguous() - x_hat.contiguous())  # L1 loss
        if self.perceptual is not None:
            p_loss = self.perceptual(x.contiguous(), x_hat.contiguous())
            nll_loss = recon_loss + self.perceptual_weight * p_loss
        else:
            nll_loss = recon_loss
        nll_loss = torch.mean(nll_loss)

        # augment for input to discriminator
        if self.diffaug_policy != '':
            x_hat_pre_aug = x_hat.detach().clone()
            x_hat = DiffAugment(x_hat, policy=self.diffaug_policy)

        # update generator
        logits_fake = self.disc(x_hat)
        g_loss = -torch.mean(logits_fake)
        last_layer = self.ae.generator.blocks[-1].weight
        d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer, self.disc_weight_max)
        d_weight *= adopt_weight(1, step, self.disc_start_step)
        loss = nll_loss + d_weight * g_loss + codebook_loss

        stats["loss"] = loss
        stats["l1"] = recon_loss.mean()
        if self.perceptual is not None:
            stats["perceptual"] = p_loss.mean()
        stats["nll_loss"] = nll_loss
        stats["g_loss"] = g_loss
        stats["d_weight"] = d_weight
        stats["codebook_loss"] = codebook_loss
        stats["latent_ids"] = quant_stats["min_encoding_indices"].squeeze(1).reshape(x.shape[0], -1)

        if "mean_distance" in stats:
            stats["mean_code_distance"] = quant_stats["mean_distance"]
        if step > self.disc_start_step:
            if self.diffaug_policy != '':
                logits_real = self.disc(DiffAugment(x.contiguous().detach(), policy=self.diffaug_policy))
            else:
                logits_real = self.disc(x.contiguous().detach())
            logits_fake = self.disc(x_hat.contiguous().detach())  # detach so that generator isn"t also updated
            d_loss = hinge_d_loss(logits_real, logits_fake)
            stats["d_loss"] = d_loss

        if self.diffaug_policy != '':
            x_hat = x_hat_pre_aug

        return x_hat, stats
    
    def train_discriminator_iter(self, step, x):
        stats = {}
        x_hat, codebook_loss, quant_stats = self.ae(x)
        if self.diffaug_policy != '':
            logits_real = self.disc(DiffAugment(x, policy=self.diffaug_policy))
            logits_fake = self.disc(DiffAugment(x_hat.contiguous().detach(), policy=self.diffaug_policy))
        else:
            logits_real = self.disc(x)
            logits_fake = self.disc(x_hat.contiguous().detach()) # detach so that generator isn"t also updated
        d_loss = hinge_d_loss(logits_real, logits_fake)
        stats["d_loss"] = d_loss
        stats["codebook_loss"] = codebook_loss
        stats["latent_ids"] = quant_stats["min_encoding_indices"].squeeze(1).reshape(x.shape[0], -1)
        if "mean_distance" in stats:
            stats["mean_code_distance"] = quant_stats["mean_distance"]

        return x_hat, stats

    def train_generator_iter(self, step, x, x_hat=None):
        stats = {}
        # update gumbel softmax temperature based on step. Anneal from 1 to 1/16 over 150000 steps
        if self.ae.quantizer_type == "gumbel":
            self.ae.quantize.temperature = max(1/16, ((-1/160000) * step) + 1)
            stats["gumbel_temp"] = self.ae.quantize.temperature

        if x_hat is None:
            x_hat, codebook_loss, quant_stats = self.ae(x)

            stats["codebook_loss"] = codebook_loss
            stats["latent_ids"] = quant_stats["min_encoding_indices"].squeeze(1).reshape(x.shape[0], -1)

            if "mean_distance" in stats:
                stats["mean_code_distance"] = quant_stats["mean_distance"]

        # get recon/perceptual loss
        recon_loss = torch.abs(x.contiguous() - x_hat.contiguous())  # L1 loss
        if self.perceptual is not None:
            p_loss = self.perceptual(x.contiguous(), x_hat.contiguous())
            nll_loss = recon_loss + self.perceptual_weight * p_loss
        else:
            nll_loss = recon_loss
        nll_loss = torch.mean(nll_loss)

        # augment for input to discriminator
        if self.diffaug_policy != '':
            x_hat_pre_aug = x_hat.detach().clone()
            x_hat = DiffAugment(x_hat, policy=self.diffaug_policy)

        # update generator
        logits_fake = self.disc(x_hat)
        g_loss = -torch.mean(logits_fake)
        last_layer = self.ae.generator.blocks[-1].weight
        d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer, self.disc_weight_max)
        d_weight *= adopt_weight(1, step, self.disc_start_step)
        loss = nll_loss + d_weight * g_loss + codebook_loss

        stats["loss"] = loss
        stats["l1"] = recon_loss.mean()
        if self.perceptual is not None:
            stats["perceptual"] = p_loss.mean()
        stats["nll_loss"] = nll_loss
        stats["g_loss"] = g_loss
        stats["d_weight"] = d_weight
        
        if self.diff_aug:
            x_hat = x_hat_pre_aug
        
        return x_hat, stats



    @torch.no_grad()
    def val_iter(self, x, step):
        stats = {}
        # update gumbel softmax temperature based on step. Anneal from 1 to 1/16 over 150000 steps
        if self.ae.quantizer_type == "gumbel":
            self.ae.quantize.temperature = max(1/16, ((-1/160000) * step) + 1)
            stats["gumbel_temp"] = self.ae.quantize.temperature

        x_hat, codebook_loss, quant_stats = self.ae(x)

        # get recon/perceptual loss
        recon_loss = torch.abs(x.contiguous() - x_hat.contiguous())  # L1 loss
        if self.perceptual is not None:
            p_loss = self.perceptual(x.contiguous(), x_hat.contiguous())
            nll_loss = recon_loss + self.perceptual_weight * p_loss
        else:
            nll_loss = recon_loss
        nll_loss = torch.mean(nll_loss)

        # update generator
        logits_fake = self.disc(x_hat)
        g_loss = -torch.mean(logits_fake)

        stats["val_l1"] = recon_loss.mean()
        if self.perceptual is not None:
            stats["val_perceptual"] = p_loss.mean()
        stats["val_nll_loss"] = nll_loss
        stats["val_g_loss"] = g_loss
        stats["val_codebook_loss"] = codebook_loss
        stats["val_latent_ids"] = quant_stats["min_encoding_indices"].squeeze(1).reshape(x.shape[0], -1)

        return x_hat, stats

    def probabilistic(self, x):
        stats = {}

        mu, logsigma, quant_stats = self.ae.probabilistic(x)
        recon = 0.5 * torch.exp(2*torch.log(torch.abs(x - mu)) - 2*logsigma)
        if torch.isnan(recon.mean()):
            log("nan detected in probabilsitic VQGAN")
        nll = recon + logsigma + 0.5*np.log(2*np.pi)
        stats['nll'] = nll.mean(0).sum() / (np.log(2) * np.prod(x.shape[1:]))
        stats['nll_raw'] = nll.sum((1, 2, 3))
        stats['latent_ids'] = quant_stats['min_encoding_indices'].squeeze(1).reshape(x.shape[0], -1)
        x_hat = mu + 0.5*torch.exp(logsigma)*torch.randn_like(logsigma)

        return x_hat, stats

class VQAutoEncoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.quantizer_type = H.model.quantizer
        self.encoder = Encoder(
            H.data.channels,
            H.model.nf,
            H.model.emb_dim,
            H.model.ch_mult,
            H.model.res_blocks,
            H.data.img_size,
            H.model.attn_resolutions
        )
        if H.model.quantizer == "nearest":
            self.quantize = VectorQuantizer(
                H.model.codebook_size, 
                H.model.emb_dim, 
                H.model.beta
            )
        elif H.model.quantizer == "gumbel":
            self.quantize = GumbelQuantizer(
                H.model.codebook_size,
                H.model.emb_dim,
                H.model.emb_dim,
                H.model.gumbel_straight_through,
                H.model.gumbel_kl_weight
            )
        self.generator = Generator(
            H.model.emb_dim, 
            H.data.channels, 
            H.model.nf, 
            H.model.ch_mult, 
            H.model.res_blocks, 
            H.data.img_size, 
            H.model.attn_resolutions
        )

    def forward(self, x):
        x = self.encoder(x)
        quant, codebook_loss, quant_stats = self.quantize(x)
        x = self.generator(quant)
        return x, codebook_loss, quant_stats

    def probabilistic(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            quant, _, quant_stats = self.quantize(x)
        mu, logsigma = self.generator.probabilistic(quant)
        return mu, logsigma, quant_stats

class Encoder(nn.Module):
    def __init__(self, in_channels, nf, out_channels, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        num_resolutions = len(ch_mult)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        blocks = []
        # initial convoltion
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nf, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        num_resolutions = len(ch_mult)
        block_in_ch = nf * ch_mult[-1]
        curr_res = resolution // 2 ** (num_resolutions-1)

        blocks = []
        # initial conv
        blocks.append(nn.Conv2d(in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(num_resolutions)):
            block_out_ch = nf * ch_mult[i]

            for _ in range(num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)

        # used for calculating ELBO - fine tuned after training
        self.logsigma = nn.Sequential(
                            nn.Conv2d(block_in_ch, block_in_ch, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(block_in_ch, in_channels, kernel_size=1, stride=1, padding=0)
                        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def probabilistic(self, x):
        with torch.no_grad():
            for block in self.blocks[:-1]:
                x = block(x)
            mu = self.blocks[-1](x)
        logsigma = self.logsigma(x)
        return mu, logsigma

# patch based discriminator
class Discriminator(nn.Module):
    def __init__(self, nc, ndf, n_layers=3):
        super().__init__()

        layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ndf_mult = 1
        ndf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        layers += [
            nn.Conv2d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# Define VQVAE classes
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta  # commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        mean_distance = torch.mean(d)

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
            "mean_distance": mean_distance
            }

    def get_codebook_entry(self, indices, shape):
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:  # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        return z_q


class GumbelQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, straight_through=False, kl_weight=5e-4, temp_init=1.0):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.proj = nn.Conv2d(num_hiddens, codebook_size, 1)  # projects last encoder layer to quantized logits
        self.embed = nn.Embedding(codebook_size, emb_dim)

    def forward(self, z):
        hard = self.straight_through if self.training else True

        logits = self.proj(z)

        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)

        z_q = torch.einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)

        qy = F.softmax(logits, dim=1)

        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()

        min_encoding_indices = soft_one_hot.argmax(dim=1)

        return z_q, diff, {
            "min_encoding_indices": min_encoding_indices
        }


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_
