import torch
from einops import rearrange
from torch.utils.data import DataLoader
import numpy as np
from prdc import compute_prdc

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
from tqdm import tqdm

from models.vqgan_3d import Generator as Generator3D
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, get_latent_loaders, get_sampler, latent_ids_to_onehot
from utils.log_utils import load_model, log
from utils.visual import to_unNorm, back_to_HU, save_volume
from utils.metrics import Structural_Similarity, Peak_Signal_to_Noise_Rate, MAE, MSE
from evaluate.feature_extractor import FeatureExtractor3D

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("ct_config", "configs/default_ct_256_vqgan_config.py", "CT VQGAN training configuration.", lock_config=True)
config_flags.DEFINE_config_file("xray_config", "configs/default_xray_vqgan_config.py", "XRay VQGAN training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

# Torch options
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = torch.device('cuda')


def get_samples(H, n_samples, loader, sampler, generator):
    imgs = []
    for _ in range(n_samples):
        latent_batch = next(loader)
        latent_batch = latent_batch["xray_embed"].to(device, non_blocking=True)
        latent_batch = rearrange(latent_batch, "b () r l c -> b (r l) c")
        x_sampled = sampler.sample(context=latent_batch[:H.diffusion.sampling_batch_size], sample_steps=H.diffusion.sampling_steps, temp=H.diffusion.sampling_temp, train=False)
        imgs.append(reconstruct_from_codes(H, sampler, x_sampled, generator))
    return imgs


def calc_prdc_from_loaders(H, latent_loader, data_loader, sampler, generator, extractor):
    batch_precision = []
    batch_recall = []
    batch_densities = []
    batch_coverages = []
    nll_loss = []
    lat_loader = iter(latent_loader)

    for real_batch in tqdm(data_loader):
        with torch.no_grad():
            real_batch = real_batch["ct"].to(device, non_blocking=True)
            fake_batch = get_samples(H, len(real_batch), lat_loader, sampler, generator)
            fake_batch = torch.cat(fake_batch, dim=0)

            recon_loss = torch.abs(real_batch - fake_batch)
            nll = torch.mean(recon_loss)
            nll_loss.append(nll)

            real_feats = extractor(real_batch).cpu().numpy()
            fake_feats = extractor(fake_batch).cpu().numpy()

            metrics = compute_prdc(
                real_features=real_feats,
                fake_features=fake_feats,
                nearest_k=3
            )

            batch_precision.append(metrics['precision'])
            batch_recall.append(metrics['recall'])
            batch_densities.append(metrics['density'])
            batch_coverages.append(metrics['coverage'])

    nll = sum(nll_loss) / len(nll_loss)
    precision = sum(batch_precision) / len(batch_precision)
    recall = sum(batch_recall) / len(batch_recall)
    density = sum(batch_densities) / len(batch_densities)
    coverage = sum(batch_coverages) / len(batch_coverages)

    print(f"NLL: {nll}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Density: {density}")
    print(f"Coverage: {coverage}")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def reconstruct_from_codes(H, sampler, x, generator):
    latents_one_hot = latent_ids_to_onehot(x, H.ct_config.model.latent_shape, H.ct_config.model.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())
    return images


def main(argv):
    H = FLAGS.config
    H.ct_config = FLAGS.ct_config
    H.xray_config = FLAGS.xray_config

    # get features from original dataset
    if H.data.dataset == 'bags':
        from utils.dataloader import BagXCT_dataset
        test_dataset = BagXCT_dataset(data_dir=H.data.data_dir, train=False, 
                                       xray_scale=H.xray_config.data.img_size, 
                                       ct_scale=H.ct_config.data.img_size,
                                       direction='both',
                                       types='grayscale',    
                                       load_res=H.data.load_res,
                                       cupy=H.data.cupy)
    else:
        from utils.dataloader import XCT_dataset
        test_dataset = XCT_dataset(data_dir=H.data.data_dir, train=False, 
                                   xray_scale=H.xray_config.data.img_size, 
                                   ct_scale=H.ct_config.data.img_size ,
                                   projections=H.data.num_xrays, 
                                   load_res=H.data.load_res,
                                   dataset=H.data.dataset,
                                   cupy=H.data.cupy,
                                   use_synthetic=H.data.use_synthetic)

        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, 
                                 num_workers=0, pin_memory=True, drop_last=True)

    # Read latents
    latents_filepath = f'logs/{H.run.name}_{H.run.experiment}/train_latents'
    assert os.path.exists(latents_filepath), f"Error: Latents path {latents_filepath} not found"

    # Load latents
    val_latents_fp = f'logs/{H.run.name}_{H.run.experiment}/val_latents'
    print(f"_| Loading latents: {val_latents_fp}")
    val_latent_ids = torch.load(val_latents_fp)
    test_latent_loader = torch.utils.data.DataLoader(val_latent_ids, batch_size=1, shuffle=False)

    # Load CT Generator
    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H.ct_config, ['quantize', 'generator'], remove_component_from_key=True)
    ct_embedding_weight = quanitzer_and_generator_state_dict.pop('embedding.weight').to(device)
    generator_ct = Generator3D(
        H.ct_config.model.emb_dim, 
        H.ct_config.data.channels, 
        H.ct_config.model.nf, 
        H.ct_config.model.ch_mult, 
        H.ct_config.model.res_blocks, 
        H.ct_config.data.img_size, 
        H.ct_config.model.attn_resolutions
    )
    generator_ct.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator_ct = generator_ct.to(device)

    # Create and load latent sampler
    sampler = get_sampler(H, ct_embedding_weight).to(device)
    sampler = load_model(sampler, H.model.name, H.model.load_step, f'{H.run.name}_{H.run.experiment}').to(device)

    sampler = sampler.eval()

    feature_extractor = FeatureExtractor3D().to(device)
    feature_extractor.apply(weights_init)
    feature_extractor = feature_extractor.eval()

    # saving generated features
    calc_prdc_from_loaders(H, test_latent_loader, test_loader, sampler, generator_ct, feature_extractor)

if __name__ == '__main__':
    app.run(main)
