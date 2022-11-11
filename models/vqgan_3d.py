import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffaug_3d import DiffAugment, rand_cutout
from utils.vqgan_utils import normalize, swish, adopt_weight, hinge_d_loss, calculate_adaptive_weight
from utils.log_utils import log
from einops import rearrange

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def activations_difference(acts1, acts2):
    val = 0
    for act1, act2 in zip(acts1, acts2):
        act1 = normalize_tensor(act1)
        act2 = normalize_tensor(act2)
        diff = (act1 - act2) ** 2
        diff = diff.mean(dim=(1,2,3,4))
        val += diff
    return torch.mean(val)

class VQGAN(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.ae = VQAutoEncoder(H)
        self.disc = Discriminator(
            H.data.channels,
            H.model.ndf,
            n_layers=H.model.disc_layers
        )
        self.disc_start_step = H.model.disc_start_step
        self.disc_weight_max = H.model.disc_weight_max
        self.diffaug_policy = H.model.diffaug_policy
        self.recon_weight = H.model.recon_weight
        self.pre_augmentation = H.model.pre_augmentation
        self.perceptual_loss = H.model.perceptual_loss
        self.perceptual_weight = H.model.perceptual_weight

    def train_iter_together(self, x, step):
        stats = {}
        # update gumbel softmax temperature based on step. Anneal from 1 to 1/16 over 150000 steps
        if self.ae.quantizer_type == "gumbel":
            self.ae.quantize.temperature = max(1/16, ((-1/160000) * step) + 1)
            stats["gumbel_temp"] = self.ae.quantize.temperature
        
        if self.pre_augmentation:
            x = rand_cutout(x, ratio=0.25, apply_ratio=0.9)

        x_hat, codebook_loss, quant_stats = self.ae(x)

        # get recon loss
        recon_loss = torch.abs(x.contiguous() - x_hat.contiguous())  # L1 loss
        nll_loss = recon_loss 
        nll_loss = torch.mean(nll_loss)

        # augment for input to discriminator
        if self.diffaug_policy != '':
            x_hat_pre_aug = x_hat#.detach().clone()
            x_hat = DiffAugment(x_hat, policy=self.diffaug_policy)

        # update generator
        logits_fake, fake_activations = self.disc(x_hat)
        g_loss = -torch.mean(logits_fake)
        last_layer = self.ae.generator.blocks[-1].weight
        d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer, self.disc_weight_max)
        d_weight *= adopt_weight(1, step, self.disc_start_step)
        loss = self.recon_weight * nll_loss + d_weight * g_loss + codebook_loss
    
        if self.perceptual_loss:
            if self.diffaug_policy != '':
                # redo activations without augmentations
                _, fake_activations = self.disc(x_hat_pre_aug)
            # No nead to pass grads though to the discriminator to change features for ground truth
            with torch.no_grad():
                _, real_activations = self.disc(x.contiguous().detach())
            perceptual_loss = activations_difference(fake_activations, real_activations)
            loss += self.perceptual_weight * perceptual_loss
            stats["perceptual_loss"] = perceptual_loss

        stats["loss"] = loss
        stats["l1"] = recon_loss.mean()
        stats["nll_loss"] = nll_loss
        stats["g_loss"] = g_loss
        stats["d_weight"] = d_weight
        stats["codebook_loss"] = codebook_loss
        stats["latent_ids"] = quant_stats["min_encoding_indices"].squeeze(1).reshape(x.shape[0], -1)

        if "mean_distance" in stats:
            stats["mean_code_distance"] = quant_stats["mean_distance"]
        if step > self.disc_start_step:
            if self.diffaug_policy != '':
                logits_real, _ = self.disc(DiffAugment(x.contiguous().detach(), policy=self.diffaug_policy))
            else:
                logits_real = self.disc(x.contiguous().detach())
            logits_fake, _ = self.disc(x_hat.contiguous().detach())  # detach so that generator isn"t also updated
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
        loss = self.recon_weight * nll_loss + d_weight * g_loss + codebook_loss

        stats["loss"] = loss
        stats["l1"] = recon_loss.mean()
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
        nll_loss = recon_loss
        nll_loss = torch.mean(nll_loss)

        # update generator
        logits_fake = self.disc(x_hat)
        g_loss = -torch.mean(logits_fake)

        stats["val_l1"] = recon_loss.mean()
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
            H.model.attn_resolutions,
            H.model.resblock_name
        )
        if H.model.quantizer == "nearest":
            self.quantize = VectorQuantizer(
                H.model.codebook_size, 
                H.model.emb_dim, 
                H.model.beta
            )
        elif H.model.quantizer == "gumbel":
            raise Exception("Gumbel Quantized not yet implemented for 3D inputs")
        self.generator = Generator(
            H.model.emb_dim, 
            H.data.channels, 
            H.model.nf, 
            H.model.ch_mult, 
            H.model.res_blocks, 
            H.data.img_size, 
            H.model.attn_resolutions,
            H.model.resblock_name
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
    def __init__(self, in_channels, nf, out_channels, ch_mult, num_res_blocks, resolution, attn_resolutions, resblock_name):
        super().__init__()
        num_resolutions = len(ch_mult)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        blocks = []
        # initial convoltion
        blocks.append(nn.Conv3d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        resblocks = {
            "resblock": ResBlock,
            "depthwise_block": DepthwiseResBlock
        }
        Block = resblocks[resblock_name]

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(num_res_blocks):
                blocks.append(Block(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # non-local attention block
        blocks.append(Block(block_in_ch, block_in_ch))
        if curr_res in attn_resolutions:
            blocks.append(AttnBlock(block_in_ch))
        blocks.append(Block(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv3d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nf, ch_mult, num_res_blocks, resolution, attn_resolutions, resblock_name):
        super().__init__()
        num_resolutions = len(ch_mult)
        block_in_ch = nf * ch_mult[-1]
        curr_res = resolution // 2 ** (num_resolutions-1)

        blocks = []
        # initial conv
        blocks.append(nn.Conv3d(in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        resblocks = {
            "resblock": ResBlock,
            "depthwise_block": DepthwiseResBlock
        }
        Block = resblocks[resblock_name]


        # non-local attention block
        blocks.append(Block(block_in_ch, block_in_ch))
        if curr_res in attn_resolutions:
            blocks.append(AttnBlock(block_in_ch))
        blocks.append(Block(block_in_ch, block_in_ch))

        for i in reversed(range(num_resolutions)):
            block_out_ch = nf * ch_mult[i]

            for _ in range(num_res_blocks):
                blocks.append(Block(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv3d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)

        # used for calculating ELBO - fine tuned after training
        self.logsigma = nn.Sequential(
                            nn.Conv3d(block_in_ch, block_in_ch, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv3d(block_in_ch, in_channels, kernel_size=1, stride=1, padding=0)
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

        layers = [nn.Conv3d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ndf_mult = 1
        ndf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv3d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv3d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        layers += [
            nn.Conv3d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        activations = []
        for layer in self.main:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                activations.append(x)
        return x, activations

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
        batch_size, _, height, width, depth = z.shape
        # reshape z -> (batch, height, width, depth, channel) and flatten
        z_flattened = rearrange(z, "b c h w d -> (b h w d) c").contiguous()

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        mean_distance = torch.mean(d)

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight)
        z_q = rearrange(z_q, "(b h w d) c -> b c h w d", b=batch_size, h=height, w=width, d=depth).contiguous()
        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape

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
            z_q = z_q.view(shape).permute(0, 4, 1, 2, 3).contiguous()

        return z_q
    
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

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
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

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

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class DepthwiseResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = LayerNorm(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.norm2 = LayerNorm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.conv_out = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        
        return x + x_in

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalize(in_channels)
        self.qkv = nn.Linear(in_channels, in_channels*3)
        self.proj_out = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        x_in = x
        x = self.norm(x)
        _, c, h, w, d = x.shape
        x = rearrange(x, "b c h w d -> b (h w d) c")
        q, k, v = self.qkv(x).chunk(3, dim=2)

        # compute attention
        attn = (q @ k.transpose(-2, -1)) * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = self.proj_out(out)
        out = rearrange(out, "b (h w d) c -> b c h w d", h=h, w=w, d=d)

        return x_in + out