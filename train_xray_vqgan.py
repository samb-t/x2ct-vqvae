import torch
from torch.utils.data import DataLoader
from einops import rearrange
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

from models.vqgan_2d import VQGAN
from utils.log_utils import flatten_collection, track_variables, log_stats, plot_images, save_model, config_log
from utils.train_utils import optim_warmup, update_ema
from utils.vqgan_utils import load_vqgan_from_checkpoint


# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
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

def train(H, vqgan, vqgan_ema, train_loader, test_loader, optim, d_optim, start_step, vis=None):
    scaler, d_scaler = None, None
    if H.train.amp:
        scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()
    
    global_step = start_step
    tracked_stats = defaultdict(lambda: np.array([]))
    tracked_stats["latent_ids"] = []
    test_tracked_stats = defaultdict(lambda: np.array([]))
    while global_step <= H.train.total_steps:
        for it, data in enumerate(train_loader):
            x = data["xray"]
            start_time = time.time()

            if global_step < H.optimizer.warmup_steps:
                optim_warmup(global_step, optim, H.optimizer.learning_rate, H.optimizer.warmup_steps)

            global_step += 1
            x = x.to(device, non_blocking=True)

            if H.train.gan_training_mode == "together":
                with torch.cuda.amp.autocast(enabled=H.train.amp):
                    x_hat, stats = vqgan.train_iter_together(x, global_step)
                
                # Update generator
                update_model_weights(optim, stats['loss'], amp=H.train.amp, scaler=scaler)
                # Update discriminator
                if global_step > H.model.disc_start_step:
                    update_model_weights(d_optim, stats['d_loss'], amp=H.train.amp, scaler=d_scaler)
                    # Adjusting APA from discriminator
                    # if (it + 1) % vqgan.APA.it == 0 and global_step >= vqgan.APA.start:
                    #     vqgan.APA.adjust_prob(H.train.batch_size)
            
            elif H.train.gan_training_mode == "alternating":
                # Update discriminator
                x_hat, d_stats = None, dict()
                if global_step > H.model.disc_start_step:
                    with torch.cuda.amp.autocast(enabled=H.train.amp):
                        x_hat, d_stats = vqgan.train_discriminator_iter(global_step, x)
                    update_model_weights(d_optim, d_stats['d_loss'], amp=H.train.amp, scaler=d_scaler)
                    # Adjusting APA from discriminator
                    if (it + 1) % vqgan.APA.it == 0 and global_step >= vqgan.APA.start:
                        vqgan.APA.adjust_prob(H.train.batch_size)
                
                # Update generator
                with torch.cuda.amp.autocast(enabled=H.train.amp):
                    x_hat, stats = vqgan.train_generator_iter(global_step, x, x_hat=x_hat)
                update_model_weights(optim, stats['loss'], amp=H.train.amp, scaler=scaler)

                # Merge stats dict
                stats.update(d_stats)
            
            else:
                raise Exception("Unknown option set for 'config.train.gan_training_mode'")
            
            if global_step % H.train.ema_update_every == 0:
                update_ema(vqgan, vqgan_ema, H.train.ema_decay)

            # Track variables
            stats["step_time"] = time.time() - start_time
            track_variables(tracked_stats, stats)
            tracked_stats["latent_ids"].append(stats['latent_ids'].cpu().contiguous())

            wandb_dict = dict()
            ## Plot graphs
            # Averages tracked variables, prints, and graphs on wandb
            if global_step % H.train.plot_graph_steps == 0 and global_step > 0:
                wandb_dict.update(log_stats(H, global_step, tracked_stats, log_to_file=H.run.log_to_file))
            
            ## Plot recons
            if global_step % H.train.plot_recon_steps == 0 and global_step > 0:
                wandb_dict.update(plot_images(H, x, title='x', vis=vis))
                wandb_dict.update(plot_images(H, x_hat, title='x_recon', vis=vis))
            
            ## Evaluate on test set
            if global_step % H.train.eval_steps == 0 and global_step > 0:
                for test_data in test_loader:
                    test_x = test_data["xray"].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=H.train.amp):
                        test_x_hat, test_stats = vqgan.val_iter(test_x, global_step)
                    track_variables(test_tracked_stats, test_stats)
                wandb_dict.update(log_stats(H, global_step, test_tracked_stats, test=True, log_to_file=H.run.log_to_file))
                wandb_dict.update(plot_images(H, test_x, title='test_x', vis=vis))
                wandb_dict.update(plot_images(H, test_x_hat, title='test_x_hat', vis=vis))
            
            ## Plot everything to wandb
            if wandb_dict:
                wandb.log(wandb_dict, step=global_step)
            
            ## Checkpoint
            if global_step % H.train.checkpoint_steps == 0 and global_step > 0:
                save_model(vqgan, 'vqgan', global_step, f"{H.run.name}_{H.run.experiment}")
                save_model(optim, 'ae_optim', global_step, f"{H.run.name}_{H.run.experiment}")
                save_model(d_optim, 'disc_optim', global_step, f"{H.run.name}_{H.run.experiment}")
                save_model(vqgan_ema, 'vqgan_ema', global_step, f"{H.run.name}_{H.run.experiment}")
            


def main(argv):
    H = FLAGS.config
    train_kwargs = {}

    # wandb can be disabled by passing in --config.run.wandb_mode=disabled
    wandb.init(name=H.run.experiment, project=H.run.name, config=flatten_collection(H), save_code=True, dir=H.run.wandb_dir, mode=H.run.wandb_mode)
    if H.run.enable_visdom:
        train_kwargs['vis'] = visdom.Visdom(server=H.run.visdom_server, port=H.run.visdom_port)
    
    if H.run.log_to_file:
        config_log(H.run.name)

    vqgan = VQGAN(H)
    vqgan_ema = copy.deepcopy(vqgan)

    print(f"Number of parameters: {sum(p.numel() for p in vqgan.parameters())}")

    vqgan = vqgan.to(device)
    vqgan_ema = vqgan_ema.to(device)

    if H.data.dataset == 'bags':
        from utils.dataloader import BagXRay_dataset
        train_dataset = BagXRay_dataset(data_dir=H.data.data_dir, train=True,
                                        scale=H.data.img_size, 
                                        types=H.data.types,
                                        direction=H.data.direction,
                                        degrees=H.data.degrees)
        test_dataset = BagXRay_dataset(data_dir=H.data.data_dir, train=False,
                                       scale=H.data.img_size, 
                                       types=H.data.types,
                                       direction=H.data.direction,
                                       degrees=H.data.degrees)
    elif H.data.dataset == 'shrec16':
        from utils.dataloader import Dataset2D
        train_dataset = Dataset2D(data_dir=H.data.data_dir, train=True)
        test_dataset = Dataset2D(data_dir=H.data.data_dir, train=False)
    elif H.data.dataset == 'chest':
        from utils.dataloader import XRay_dataset
        train_dataset = XRay_dataset(data_dir=H.data.data_dir, train=True, scale=H.data.img_size, 
                                     projections=H.data.num_xrays, dataset=H.data.dataset,
                                     use_synthetic=H.data.use_synthetic)
        test_dataset = XRay_dataset(data_dir=H.data.data_dir, train=False, scale=H.data.img_size, 
                                projections=H.data.num_xrays, dataset=H.data.dataset,
                                use_synthetic=H.data.use_synthetic)
    else:
        raise Exception("Dataset not supported!")
    workers = 2 if H.train.batch_size < 32 else 4

    train_loader = DataLoader(train_dataset, batch_size=H.train.batch_size, shuffle=True, 
                              num_workers=workers, pin_memory=True, drop_last=True)

    # TODO: Fix evaluation step to work with different sized batches so we can set drop_last here to False
    test_loader = DataLoader(test_dataset, batch_size=H.train.test_batch_size, shuffle=True, 
                             num_workers=workers, pin_memory=True, drop_last=True) 
    
    optim = torch.optim.Adam(vqgan.ae.parameters(), lr=H.optimizer.learning_rate)
    d_optim = torch.optim.Adam(vqgan.disc.parameters(), lr=H.optimizer.learning_rate)

    start_step = 0
    if H.train.load_step > 0:
        start_step = H.train.load_step + 1  # don't repeat the checkpointed step
        vqgan, optim, d_optim, vqgan_ema = load_vqgan_from_checkpoint(H, vqgan, optim, d_optim, vqgan_ema)

    train(H, vqgan, vqgan_ema, train_loader, test_loader, optim, d_optim, start_step, **train_kwargs)


if __name__ == '__main__':
    app.run(main)
