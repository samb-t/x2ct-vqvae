import logging
import numpy as np
import os
import torch
import torchvision
import visdom
from ml_collections import ConfigDict
import wandb

def config_log(log_dir, filename="log.txt"):
    log_dir = "logs/" + log_dir
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, filename),
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )


def log(output, log_to_file=False):
    if log_to_file:
        logging.info(output)
    print(output)


# Tracking functions
def track_variables(logs, stats):
    for name, variable in stats.items():
        if isinstance(variable, float):
            logs['track_'+name] = np.append(logs['track_'+name], variable)
        elif variable.numel() == 1:
            logs['track_'+name] = np.append(logs['track_'+name], variable.item())

def update_logs_means(logs):
    for name in list(logs.keys()):
        if 'track_' in name:
            mean_name = 'mean_'+name[6:]
            logs[mean_name] = np.mean(logs[name])
            logs[name] = np.array([])

def log_stats(H, step, stats, test=False, log_to_file=False):
    # Calculate means
    update_logs_means(stats)
    # Clean stats
    print_stats = {key.replace('mean_',''): value for key, value in stats.items() if isinstance(value, float) and 'mean_' in key}
    log_str = f"Step: {step}  " if not test else f"TEST STATS - Step: {step}  "
    for stat, val in print_stats.items():
        log_str += f"{stat}: {val:.4f}  "
        # TODO: Enable visdom plotting
        if H.run.enable_visdom:
            ...
    log(log_str, log_to_file=log_to_file)
    return print_stats

def plot_images(H, x, title='', norm=False, vis=None):
    # x = (x + 1) / 2 if norm else x
    x = torch.clamp(x, -1, 1)
    # x = (x - x.min()) / (x.max() - x.min())

    # visdom
    if H.run.enable_visdom and vis is not None:
        vis.images(x, win=title, opts=dict(title=title))
    
    # wandb
    x = wandb.Image(x, caption=title)
    return {title: x}


def start_training_log(hparams):
    log("Using following hparams:")
    param_keys = list(hparams)
    param_keys.sort()
    for key in param_keys:
        log(f"> {key}: {hparams[key]}")


def save_model(model, model_save_name, step, log_dir):
    log_dir = "logs/" + log_dir + "/saved_models"
    os.makedirs(log_dir, exist_ok=True)
    model_name = f"{model_save_name}_{step}.th"
    log(f"Saving {model_save_name} to {model_save_name}_{str(step)}.th under {log_dir}")
    torch.save(model.state_dict(), os.path.join(log_dir, model_name))


def load_model(model, model_load_name, step, log_dir, strict=False):
    log(f"Loading {model_load_name}_{str(step)}.th from {log_dir}")
    log_dir = "logs/" + log_dir + "/saved_models"
    try:
        model.load_state_dict(
            torch.load(os.path.join(log_dir, f"{model_load_name}_{str(step)}.th")),
            strict=strict,
        )
    except TypeError:  # for some reason optimisers don't like the strict keyword
        model.load_state_dict(
            torch.load(os.path.join(log_dir, f"{model_load_name}_{str(step)}.th")),
        )

    return model


def display_images(vis, images, H, win_name=None):
    if win_name is None:
        win_name = f"{H.model}_images"
    images = torchvision.utils.make_grid(images.clamp(0, 1), nrow=int(np.sqrt(images.size(0))), padding=0)
    vis.image(images, win=win_name, opts=dict(title=win_name))


def save_images(images, im_name, step, log_dir, save_individually=False):
    log_dir = "logs/" + log_dir + "/images"
    os.makedirs(log_dir, exist_ok=True)
    if save_individually:
        for idx in range(len(images)):
            torchvision.utils.save_image(torch.clamp(images[idx], 0, 1), f"{log_dir}/{im_name}_{step}_{idx}.png")
    else:
        torchvision.utils.save_image(
            torch.clamp(images, 0, 1),
            f"{log_dir}/{im_name}_{step}.png",
            nrow=int(np.sqrt(images.shape[0])),
            padding=0
        )


def save_latents(H, train_latent_ids, val_latent_ids):
    save_dir = f'logs/{H.run.name}_{H.run.experiment}'
    os.makedirs(save_dir, exist_ok=True)

    train_latents_fp = f'logs/{H.run.name}_{H.run.experiment}/train_latents'
    val_latents_fp = f'logs/{H.run.name}_{H.run.experiment}/val_latents'

    torch.save(train_latent_ids, train_latents_fp)
    torch.save(val_latent_ids, val_latents_fp)


def save_stats(H, stats, step):
    save_dir = f"logs/{H.run.name}_{H.run.experiment}/saved_stats"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"logs/{H.run.name}_{H.run.experiment}/saved_stats/stats_{step}"
    log(f"Saving stats to {save_path}")
    torch.save(stats, save_path)


def load_stats(H, step):
    load_path = f"logs/{H.run.name}_{H.run.experiment}/saved_stats/stats_{step}"
    stats = torch.load(load_path)
    return stats


def set_up_visdom(H):
    server = H.visdom_server
    try:
        if server:
            vis = visdom.Visdom(server=server, port=H.visdom_port)
        else:
            vis = visdom.Visdom(port=H.visdom_port)
        return vis

    except Exception:
        log_str = "Failed to set up visdom server - aborting"
        log(log_str, level="error")
        raise RuntimeError(log_str)

def flatten_collection(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(flatten_collection(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
