import torch
from einops import rearrange
from torch.utils.data import DataLoader
import numpy as np

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
from tqdm import tqdm

from models.vqgan_3d import Generator as Generator3D
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, get_latent_loaders, get_sampler, latent_ids_to_onehot
from utils.log_utils import load_model
from utils.visual import to_unNorm, back_to_HU, save_volume
from utils.metrics import MSE
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

def evaluate(H, sampler, generator_ct, test_loader, test_real_loader, res_dir):              
    mse_total_avg = 0.
    feature_extractor = FeatureExtractor3D().to(device)
    feature_extractor.apply(weights_init).eval()
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for it, test_data in pbar:
        real_data = next(test_real_loader)
        test_real = real_data["ct"]
        test_x = test_data["ct_codes"].to(device, non_blocking=True)
        test_context = test_data["xray_embed"].to(device, non_blocking=True)
        test_context = rearrange(test_context, "b () r l c -> b (r l) c")
        test_x = rearrange(test_x, "b () l -> b l")
        test_real = reconstruct_from_codes(H, sampler, test_x[:H.diffusion.sampling_batch_size], generator_ct)
        x_sampled = sampler.sample(context=test_context[:H.diffusion.sampling_batch_size], sample_steps=H.diffusion.sampling_steps, temp=H.diffusion.sampling_temp, train=False)
        x_sampled_img = reconstruct_from_codes(H, sampler, x_sampled, generator_ct)

        # tensor -> numpy array -> unnormalisation
        real_ct, recon_ct = to_unNorm(test_real[0], x_sampled_img[0])

        mse = MSE(real_ct, recon_ct, size_average=False)

        if H.data.dataset != 'bags':
            # back to Hounsfield scale for visualisation
            real_ct = back_to_HU(real_ct).astype(np.int32) - 1024
            recon_ct = back_to_HU(recon_ct).astype(np.int32) - 1024
        else:
            real_ct *= 4095
            recon_ct *= 4095

        mse_total_avg += float(mse)
        pbar.set_description(
            f"mse: {np.round(mse, 7)}")

        # Saving to .raw for visualisation
        save_volume(real_ct, os.path.join(res_dir, f"real_ct_{it:04}"))
        save_volume(recon_ct, os.path.join(res_dir,
                                           f"recon_ct_{it:04}"))

    total_mse = round((mse_total_avg/len(test_loader)), 4)

    print(f">>>> Total avg mse: {total_mse}")


def main(argv):
    H = FLAGS.config
    H.ct_config = FLAGS.ct_config
    H.xray_config = FLAGS.xray_config
    
    outputs_dir = f'eval_outputs/{H.run.name}/{str(H.model.load_step)}'
    try:
        os.makedirs(outputs_dir, exist_ok=True)
    except OSError:
        pass

    # Read latents
    latents_filepath = f'logs/{H.run.name}_{H.run.experiment}/train_latents'
    assert os.path.exists(latents_filepath), f"Error: Latents path {latents_filepath} not found"

    # Load latents
    _, test_latent_loader = get_latent_loaders(H)

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

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                             num_workers=0, pin_memory=True, drop_last=False)

    test_dataloader = iter(test_loader)

    print("LEN: ", len(test_loader))
    evaluate(H, sampler, generator_ct,
             test_latent_loader, test_dataloader, outputs_dir)


if __name__ == '__main__':
    app.run(main)
