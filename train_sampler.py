import torch
from torch.utils.data import DataLoader
from einops import rearrange, repeat
import numpy as np

import wandb
import visdom
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import time
import copy
import os
from collections import defaultdict
from tqdm import tqdm

from models.vqgan_2d import VQAutoEncoder as VQAutoEncoder2D, Generator as Generator2D
from models.vqgan_3d import VQAutoEncoder as VQAutoEncoder3D, Generator as Generator3D
from utils.dataloader import XCT_dataset, BagXCT_dataset
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, generate_latent_ids, get_latent_loaders, get_sampler, latent_ids_to_onehot
from utils.log_utils import log, flatten_collection, track_variables, log_stats, plot_images, save_model, config_log, load_model
from utils.train_utils import optim_warmup, update_ema

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

def update_model_weights(optim, loss, amp=False, scaler=None):
    optim.zero_grad()
    if amp:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()

def reconstruct_from_codes(H, sampler, x, generator):
    latents_one_hot = latent_ids_to_onehot(x, H.ct_config.model.latent_shape, H.ct_config.model.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())
    return images

def train(H, sampler, sampler_ema, generator_ct, generator_xray, train_loader, test_loader, optim, start_step, vis=None):
    scaler = None
    if H.train.amp:
        scaler = torch.cuda.amp.GradScaler()
    
    global_step = start_step
    tracked_stats = defaultdict(lambda: np.array([]))
    tracked_stats["latent_ids"] = []
    test_tracked_stats = defaultdict(lambda: np.array([]))
    while global_step <= H.train.total_steps:
        for data in train_loader:
            start_time = time.time()
            context = data["xray_embed"].to(device, non_blocking=True)
            x = data["ct_codes"].to(device, non_blocking=True)
            # TODO: Fix the latents generation function to have this there instead of here
            
            # Randomly pick some num_xrays views to train on
            indices = torch.stack([torch.from_numpy(np.random.choice(context.size(2), H.data.num_xrays, replace=False)) for _ in range(context.size(0))]).to(device)
            indices = repeat(indices, "b r -> b () r l c", l=context.size(3), c=context.size(4))
            # TODO: Don't repeat and use fancy index select thing instead
            context = torch.gather(context, 2, indices)
            context = rearrange(context, "b () r l c -> b (r l) c")
            x = rearrange(x, "b () l -> b l")

            if global_step < H.optimizer.warmup_steps:
                optim_warmup(global_step, optim, H.optimizer.learning_rate, H.optimizer.warmup_steps)

            global_step += 1

            with torch.cuda.amp.autocast(enabled=H.train.amp):
                stats = sampler.train_iter(x, context=context)
            
            update_model_weights(optim, stats["loss"], amp=H.train.amp, scaler=scaler)

            if global_step % H.train.ema_update_every == 0:
                update_ema(sampler, sampler_ema, H.train.ema_decay)
            
            stats["step_time"] = time.time() - start_time
            track_variables(tracked_stats, stats)

            wandb_dict = dict()
            ## Plot graphs
            # Averages tracked variables, prints, and graphs on wandb
            if global_step % H.train.plot_graph_steps == 0 and global_step > 0:
                wandb_dict.update(log_stats(H, global_step, tracked_stats, log_to_file=H.run.log_to_file))
            
            ## Plot recons
            if global_step % H.train.plot_recon_steps == 0 and global_step > 0:
                # TODO: Plot x-rays as well?

                # Plot original CT scans
                x_img = reconstruct_from_codes(H, sampler, x[:H.diffusion.sampling_batch_size], generator_ct)
                wandb_dict.update(plot_images(H, x_img.mean(dim=3), title='gt ct mean(dim=3)', vis=vis))
                wandb_dict.update(plot_images(H, x_img.mean(dim=4), title='gt ct mean(dim=4)', vis=vis))
                
                # Plot estimated recons
                assert H.diffusion.sampling_steps <= np.prod(H.ct_config.model.latent_shape), f"Number of sampling steps {H.diffusion.sampling_steps} must be <= the number of latent elements {np.prod(H.ct_config.model.latent_shape)}"
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=H.train.amp):
                        x_sampled = sampler.sample(context=context[:H.diffusion.sampling_batch_size], sample_steps=H.diffusion.sampling_steps, temp=H.diffusion.sampling_temp)
                x_sampled_img = reconstruct_from_codes(H, sampler, x_sampled, generator_ct)
                wandb_dict.update(plot_images(H, x_sampled_img.mean(dim=3), title='x_sampled mean(dim=3)', vis=vis))
                wandb_dict.update(plot_images(H, x_sampled_img.mean(dim=4), title='x_sampled mean(dim=4)', vis=vis))
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=H.train.amp):
                        x_maxprob = sampler.sample_max_probability(context=context[:H.diffusion.sampling_batch_size], sample_steps=H.diffusion.sampling_steps)
                x_maxprob_img = reconstruct_from_codes(H, sampler, x_maxprob, generator_ct)
                wandb_dict.update(plot_images(H, x_maxprob_img.mean(dim=3), title='x_maxprob mean(dim=3)', vis=vis))
                wandb_dict.update(plot_images(H, x_maxprob_img.mean(dim=4), title='x_maxprob mean(dim=4)', vis=vis))
                
            ## Evaluate on test set
            if global_step % H.train.eval_steps == 0 and global_step > 0:
                log("Evaluating...")
                for _ in tqdm(range(H.train.eval_repeats)):
                    for test_data in test_loader:
                        test_context = test_data["xray_embed"].to(device, non_blocking=True)
                        test_x = test_data["ct_codes"].to(device, non_blocking=True)
                        test_context = rearrange(test_context, "b () r l c -> b (r l) c")
                        test_x = rearrange(test_x, "b () l -> b l")
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(enabled=H.train.amp):
                                test_stats = sampler.train_iter(test_x, context=test_context, test=True)
                        track_variables(test_tracked_stats, test_stats)
                wandb_dict.update(log_stats(H, global_step, test_tracked_stats, test=True, log_to_file=H.run.log_to_file))

            ## Checkpoint
            if global_step % H.train.checkpoint_steps == 0 and global_step > 0:
                save_model(sampler, H.model.name, global_step, f"{H.run.name}_{H.run.experiment}")
                save_model(sampler_ema, f"{H.model.name}_ema", global_step, f"{H.run.name}_{H.run.experiment}")

            ## Plot everything to wandb
            if wandb_dict:
                wandb.log(wandb_dict, step=global_step)


def main(argv):
    H = FLAGS.config
    H.ct_config = FLAGS.ct_config
    H.xray_config = FLAGS.xray_config
    train_kwargs = {}

    # wandb can be disabled by passing in --config.run.wandb_mode=disabled
    wandb.init(name=H.run.experiment, project=H.run.name, config=flatten_collection(H), save_code=True, dir=H.run.wandb_dir, mode=H.run.wandb_mode)
    if H.run.enable_visdom:
        train_kwargs['vis'] = visdom.Visdom(server=H.run.visdom_server, port=H.run.visdom_port)
    
    if H.run.log_to_file:
        config_log(H.run.name)

    # Create latents if not already done
    latents_filepath = f'logs/{H.run.name}_{H.run.experiment}/train_latents'
    if not os.path.exists(latents_filepath):
        # Create latents
        # Load CT VQGAN
        ae_state_dict = retrieve_autoencoder_components_state_dicts(
                H.ct_config, ['encoder', 'quantize', 'generator']
            )
        ae_ct = VQAutoEncoder3D(H.ct_config)
        ae_ct.load_state_dict(ae_state_dict, strict=False)

        # Load X-Ray VQGAN
        ae_state_dict = retrieve_autoencoder_components_state_dicts(
                H.xray_config, ['encoder', 'quantize', 'generator']
            )
        ae_xray = VQAutoEncoder2D(H.xray_config)
        ae_xray.load_state_dict(ae_state_dict, strict=False)

        if H.data.loader == "bagct":
            train_dataset = BagXCT_dataset(data_dir=H.data.data_dir, train=True, 
                                           xray_scale=H.xray_config.data.img_size, 
                                           ct_scale=H.ct_config.data.img_size)
            test_dataset = BagXCT_dataset(data_dir=H.data.data_dir, train=False, 
                                          xray_scale=H.xray_config.data.img_size, 
                                          ct_scale=H.ct_config.data.img_size)
        else:
            train_dataset = XCT_dataset(data_dir=H.data.data_dir, train=True, 
                                        xray_scale=H.xray_config.data.img_size, 
                                        scale=H.ct_config.data.img_size,
                                        projections=H.data.num_xrays, 
                                        load_res=H.data.load_res)
            test_dataset = XCT_dataset(data_dir=H.data.data_dir, train=False, 
                                        xray_scale=H.xray_config.data.img_size, 
                                        scale=H.ct_config.data.img_size ,
                                        projections=H.data.num_xrays, 
                                        load_res=H.data.load_res)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, 
                                   num_workers=2, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                    num_workers=2, pin_memory=True, drop_last=False)
        
        generate_latent_ids(H, ae_ct.cuda(), ae_xray.cuda(), train_loader, test_loader)

        # Clear AEs from  memory
        del ae_ct
        del ae_xray
    
    # Load latents
    train_latent_loader, test_latent_loader = get_latent_loaders(H)

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
        H.ct_config.model.attn_resolutions,
        H.ct_config.model.resblock_name
    )
    generator_ct.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator_ct = generator_ct.to(device)

    # Load X-Ray Generator
    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H.xray_config, ['quantize', 'generator'], remove_component_from_key=True)
    generator_xray = Generator2D(
        H.xray_config.model.emb_dim, 
        H.xray_config.data.channels, 
        H.xray_config.model.nf, 
        H.xray_config.model.ch_mult, 
        H.xray_config.model.res_blocks, 
        H.xray_config.data.img_size, 
        H.xray_config.model.attn_resolutions
    )
    generator_xray.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator_xray = generator_xray.to(device)

    # Create latent sampler
    sampler = get_sampler(H, ct_embedding_weight).to(device)
    sampler_ema = copy.deepcopy(sampler).to(device)

    if H.optimizer.weight_decay > 0:
        optim = torch.optim.AdamW(sampler.parameters(), lr=H.optimizer.learning_rate, weight_decay=H.optimizer.weight_decay)
    else:
        optim = torch.optim.Adam(sampler.parameters(), lr=H.optimizer.learning_rate)

    start_step = 0
    if H.train.load_step > 0:
        start_step = H.train.load_step + 1  # don't repeat the checkpointed step
        sampler = load_model(sampler, H.model.name, H.train.load_step, f"{H.run.name}_{H.run.experiment}").to(device)
        ema_sampler = load_model(sampler_ema, f'{H.model.name}_ema', H.train.load_step, f"{H.run.name}_{H.run.experiment}")

    train(H, sampler, sampler_ema, generator_ct, generator_xray, train_latent_loader, test_latent_loader, optim, start_step)
    


if __name__ == '__main__':
    app.run(main)
