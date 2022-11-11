from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference

def get_config():
    config = ConfigDict()

    #######################################################################
    ############################# RUN CONFIG ##############################
    #######################################################################
    config.run = run = ConfigDict()
    # Name for set of experiments in wandb
    run.name = 'baggage-vqgan'
    # Creates a separate log subfolder for each experiment
    run.experiment = 'default'
    run.wandb_dir = ''
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
    data.loader = "bagct"
    data.data_dir = '/home2/datasets/baggage/durct/process_vol'
    data.img_size = FieldReference(128)
    data.load_res = 256 # data.get_ref('img_size')
    data.channels = 1

    #######################################################################
    ########################### TRAINING CONFIG ###########################
    #######################################################################
    config.train = train = ConfigDict()
    # When the generator and discriminator are updated.
    # - "together": G_{t+1} = f(G_t, D_t) and D_{t+1} = g(G_t, D_t)
    #   In Unleashing Transformers for some reason (maybe a bug?) we trained
    #   with them both updated together, stopping gradient flow with detach.
    # - "alternating": D_{t+1} = f(G_t, D_t) then G_{t+1} = g(G_t, D_{t+1})
    #   Almost always GANs are trained in this alternating fashion. Because 
    #   the discriminator is orders of magnitude faster than the generator
    #   I have set it up so the discriminator is updated first meaning that
    #   for each batch, the generator only has to be evaluated once.
    train.gan_training_mode = "together"
    # Whether to use automatic mixed precision. For VQGAN can be worse 
    # with small batches
    train.amp = True
    train.batch_size = FieldReference(8)
    # How often to plot new loss values to graphs
    train.plot_graph_steps = 100
    # How often to plot reconstruction images
    train.plot_recon_steps = 1000
    # How often to evaluate on test set
    train.eval_steps = 1000
    # How often to save checkpoints
    train.checkpoint_steps = 10000
    # How often to update ema model params (more often is better but slightly slower)
    train.ema_update_every = 10
    train.ema_decay = 0.995
    train.load_step = 0

    #######################################################################
    ############################# MODEL CONFIG ############################
    #######################################################################
    config.model = model = ConfigDict()
    # Name of architecture. Currently in ['2d_vqgan', '3d_vqgan']. ViT architecture on TODO list
    model.name = "3d_vqgan"
    # Differential Augmentation options to be put in one string split by commas. Currently in ['translation', 'cutout', 'color']
    # Think I've seen a paper showing that spatial augmentations seem to be more useful than colour augmentations
    model.diffaug_policy = ''
    # Vector Quantizer module. Currently in ['nearest', 'gumbel']
    model.quantizer = 'nearest'
    # Vector Quantizer commitment loss
    model.beta = 0.25
    # Resolutions to apply attention to. With flash attention so fast it might be worth applying to more layers
    model.attn_resolutions = []
    # Channels mults applied to nf to increase dim
    model.ch_mult = [1, 2, 4, 8]
    # Number of codes in the codebook
    model.codebook_size = 1024
    # Dimension of each code
    model.emb_dim = 256
    # Spatial size of latents
    model.latent_shape = [1, 16, 16, 16]
    # Number of layers in the discriminator. TODO: Check this against more recent papers since it is fiarly small
    model.disc_layers = 3
    # Adaptive weight limit. Found to improve stability in Unleashing Transformers
    model.disc_weight_max = 1.0
    # What step to start using the discriminator
    model.disc_start_step = 30001
    # Base number of filters in the discriminator
    model.ndf = 32
    # Base number of filters in the autoencoder
    model.nf = 32
    # Number of residual blocks per resolution
    model.res_blocks = 2
    # Gumbel Softmax quantisation options
    model.gumbel_kl_weight = 1e-8
    model.gumbel_straight_through = False
    # Whether to use perceptual loss. Can't be used for CT Scans at the minute, no perceptual net
    model.perceptual_loss = True
    model.perceptual_weight = 0.1
    model.recon_weight = 10.0
    model.resblock_name = "resblock"
    model.pre_augmentation = True

    #######################################################################
    ########################### OPTIMIZER CONFIG ##########################
    #######################################################################
    config.optimizer = optimizer = ConfigDict()
    optimizer.base_learning_rate = FieldReference(4.5e-6)
    optimizer.learning_rate = optimizer.get_ref('base_learning_rate') * train.get_ref('batch_size')
    optimizer.warmup_steps = 0

    return config

    