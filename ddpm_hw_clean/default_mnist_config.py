import ml_collections


def create_default_mnist_config():
    config = ml_collections.ConfigDict()

    # data
    data = config.data = ml_collections.ConfigDict()
    data.image_size = 32
    data.num_channels = 1
    data.centered = True
    data.batch_size = 128
    data.norm_mean = (0.5)
    data.norm_std = (0.5)

    # model
    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.nf = 32
    model.ch_mult = (1, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.dropout = 0.1
    model.resamp_with_conv = True
    model.conditional = True
    model.nonlinearity = 'swish'

    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.0
    optim.linear_warmup = 4000
    optim.lr = 2e-4
    optim.weight_decay = 0

    # training
    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 8000
    training.checkpoint_freq = 2000
    training.eval_freq = 2000
    training.snapshot_freq = 2000
    training.snapshot_batch_size = 100
    training.batch_size = 256
    training.ode_sampling = False

    training.checkpoints_folder = './ddpm_checkpoints/'

    # sde
    sde = config.sde = ml_collections.ConfigDict()
    sde.N = 1000
    sde.beta_min = 0.1
    sde.beta_max = 20

    # 2 assignment - train noisy classifier
    classifier = config.classifier = ml_collections.ConfigDict()
    classifier.training_iters = 8000
    classifier.eval_freq = 2000
    classifier.snapshot_freq = 2000
    classifier.checkpoint_freq = 2000
    classifier.checkpoint_path = './ddpm_checkpoints/classifier.pth'

    config.device = 'cuda:0'
    return config
