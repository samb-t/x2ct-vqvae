from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference

def get_config():
    config = ConfigDict()

    #######################################################################
    ############################# RUN CONFIG ##############################
    #######################################################################
    config.run = run = ConfigDict()
    # Name for set of experiments in wandb
    run.name = 'sampler'
    # Creates a separate log subfolder for each experiment
    run.experiment = 'chest'
    run.wandb_dir = 'online'
    # Set this to 'disabled' to disable wandb logging
    run.wandb_mode = 'online'
    # Enables logging to visdom
    run.enable_visdom = False
    run.visdom_server = 'http://ncc1.clients.dur.ac.uk'
    run.visdom_port = 9275
    run.log_to_file = False

    #######################################################################
    ############################# DATA CONFIG #############################
    #######################################################################
    config.data = data = ConfigDict()
    data.data_dir = '/home2/datasets/baggage/lidc'
    data.img_size = FieldReference(128)
    data.num_xrays = 2
    data.channels = 1
    data.load_res = None
    data.dataset = 'chest'
    data.cupy = False
    data.use_synthetic = False
    data.loader = "chest"

    #######################################################################
    ########################### TRAINING CONFIG ###########################
    #######################################################################
    config.train = train = ConfigDict()
    train.amp = True
    train.batch_size = FieldReference(4)
    # How often to plot new loss values to graphs
    train.plot_graph_steps = 100
    # How often to plot reconstruction images
    train.plot_recon_steps = 1
    # How often to evaluate on test set
    train.eval_steps = 5000
    # How often to save checkpoints
    train.checkpoint_steps = 5000
    # How often to update ema model params (more often is better but slightly slower)
    train.ema_update_every = 10
    train.ema_decay = 0.995
    # What model step to load
    train.load_step = 45000
    # Number of times to repeat evaluation on the evaluations set. With diffusion
    # samplers the training loss is very noisy so multipple evaluations gives a 
    # better estimate. TODO: reaplce with sampler.elbo(...)
    train.eval_repeats = 20
    train.use_context = True
    train.total_steps = 100000

    #######################################################################
    ############################# MODEL CONFIG ############################
    #######################################################################
    config.model = model = ConfigDict()
    # Name of architecture. Currently in ['absorbing', 'autoregressive'].
    model.name = "absorbing"
    # Network width
    model.n_emb = 512
    # Number of attention heads
    model.n_head = 4
    # Number of layers
    model.n_layers = 8
    # Max input size to initialise positional embeddings etc at
    model.block_size = 1024
    # Dropout params
    model.attn_pdrop = 0.5
    model.embd_pdrop = 0.5
    model.resid_pdrop = 0.5
    model.load_step = 0

    config.diffusion = diffusion = ConfigDict()
    # What loss to use. Choose from ['elbo', 'mlm', 'reweighted_elbo']
    # - ELBO: the ELBO i.e. max likelihood
    # - MLM: loss used in generative masked language models and as BERT style 
    # approaches where cross entropy is averaged over all masked elements
    # - Reweighted ELBO: a reweighting of the ELBO proposed in 
    # "Unleashing Transformers" based on the information available to the model
    # at each time step found to improve convergence and sample quality 
    diffusion.loss_type = "reweighted_elbo"
    # How to mask tokens. Choose from ['random', 'fixed']
    # - Random: Independently mask tokens with probability t/T
    # - Fixed: Mask exactly int(t/T * latent_size) tokens
    diffusion.mask_schedule = "random"
    # Approach used to sample diffusion time steps. Choose from ['uniform', 'importance']
    # - Uniform: sample time step uniformly
    # - Importance: sample time step using importance sampling
    diffusion.time_sampling = "uniform"
    # Number of steps to sample with. Max value is of the number of elements in the latents
    diffusion.sampling_steps = 512
    # Temperature to sample diffusion with
    diffusion.sampling_temp = 0.9
    # Batch size for sampling
    diffusion.sampling_batch_size = 1
    diffusion.flash_attn = True

    #######################################################################
    ########################### OPTIMIZER CONFIG ##########################
    #######################################################################
    config.optimizer = optimizer = ConfigDict()
    optimizer.learning_rate = 2e-4
    optimizer.warmup_steps = 5000
    optimizer.weight_decay = 0.1


    return config
