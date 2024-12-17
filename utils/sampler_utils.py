import os
import torch
from einops import rearrange
from tqdm import tqdm
from .log_utils import save_latents, log
from models import Transformer, AbsorbingDiffusion, AutoregressiveTransformer


def get_sampler(H, embedding_weight):
    if H.model.name == 'absorbing':
        denoise_fn = Transformer(H).cuda()
        sampler = AbsorbingDiffusion(
            H, denoise_fn, H.ct_config.model.codebook_size, embedding_weight)

    elif H.model.name == 'autoregressive':
        sampler = AutoregressiveTransformer(H, embedding_weight)

    return sampler


@torch.no_grad()
def get_samples(H, generator, sampler):

    if H.model.name == "absorbing":
        if H.sample_type == "diffusion":
            latents = sampler.sample(sample_steps=H.diffusion.sample_steps, temp=H.diffusion.temp)
        else:
            latents = sampler.sample_mlm(temp=H.diffusion.temp, sample_steps=H.diffusion.sample_steps)

    elif H.model.name == "autoregressive":
        latents = sampler.sample(H.model.temp)

    latents_one_hot = latent_ids_to_onehot(latents, H.ct_config.model.latent_shape, H.ct_config.model.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())

    return images


def latent_ids_to_onehot(latent_ids, latent_shape, codebook_size):
    min_encoding_indices = latent_ids.view(-1).unsqueeze(1)
    encodings = torch.zeros(
        min_encoding_indices.shape[0],
        codebook_size
    ).to(latent_ids.device)
    encodings.scatter_(1, min_encoding_indices, 1)
    one_hot = encodings.view(
        latent_ids.shape[0],
        latent_shape[1],
        latent_shape[2],
        latent_shape[3],
        codebook_size
    )
    return one_hot.reshape(one_hot.shape[0], -1, codebook_size)


@torch.no_grad()
def generate_latent_ids(H, ae_ct, ae_xray, train_loader, val_loader=None):
    train_latent_ids = generate_latents_from_loader(H, ae_ct, ae_xray, train_loader)
    val_latent_ids = generate_latents_from_loader(H, ae_ct, ae_xray, val_loader)

    save_latents(H, train_latent_ids, val_latent_ids)


def generate_latents_from_loader(H, ae_ct, ae_xray, dataloader):
    latent_ids = []
    for data in tqdm(dataloader):
        xrays, ct = data["xrays"].cuda(), data["ct"].cuda()

        # NOTE: Not using AMP here to get more accurate results

        xray_latents = rearrange(xrays, "b r c h w -> (b r) c h w")
        xray_latents = ae_xray.encoder(xray_latents)
        xray_quant, _, _ = ae_xray.quantize(xray_latents)
        xray_quant = rearrange(xray_quant, "(b r) c h w -> b r (h w) c", b=xrays.size(0))
        # TODO: This can also be saved as min_encoding_indices to save memory,
        # and decompressed when training. Reltively minor since the datasets are small

        ct_latents = ae_ct.encoder(ct)
        _, _, ct_quant_stats = ae_ct.quantize(ct_latents)
        ct_min_encoding_indices = ct_quant_stats["min_encoding_indices"]
        ct_min_encoding_indices = ct_min_encoding_indices.view(ct.size(0), -1)

        latent_ids.append({
            "xray_embed": xray_quant.cpu().contiguous(),
            "ct_codes": ct_min_encoding_indices.cpu().contiguous()
        })

    return latent_ids


@torch.no_grad()
def get_latent_loaders(H, shuffle=True):
    train_latents_fp = f'logs/{H.run.name}_{H.run.experiment}/train_latents'
    val_latents_fp = f'logs/{H.run.name}_{H.run.experiment}/val_latents'

    train_latent_ids = torch.load(train_latents_fp)
    train_latent_loader = torch.utils.data.DataLoader(train_latent_ids, batch_size=H.train.batch_size, shuffle=shuffle)

    val_latent_ids = torch.load(val_latents_fp)
    val_latent_loader = torch.utils.data.DataLoader(val_latent_ids, batch_size=H.train.batch_size, shuffle=shuffle)

    return train_latent_loader, val_latent_loader


# TODO: rethink this whole thing - completely unnecessarily complicated
def retrieve_autoencoder_components_state_dicts(H, components_list, remove_component_from_key=False):
    state_dict = {}
    # default to loading ema models first
    ae_load_path = f"logs/{H.run.name}_{H.run.experiment}/saved_models/vqgan_ema_{H.model.sampler_load_step}.th"
    if not os.path.exists(ae_load_path):
        f"logs/{H.run.name}_{H.run.experiment}/saved_models/vqgan_{H.model.sampler_load_step}.th"
    log(f"Loading VQGAN from {ae_load_path}")
    full_vqgan_state_dict = torch.load(ae_load_path, map_location="cpu")

    for key in full_vqgan_state_dict:
        for component in components_list:
            if component in key:
                new_key = key[3:]  # remove "ae."
                if remove_component_from_key:
                    new_key = new_key[len(component)+1:]  # e.g. remove "quantize."

                state_dict[new_key] = full_vqgan_state_dict[key]

    return state_dict
