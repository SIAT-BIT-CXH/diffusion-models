from default_mnist_config import create_default_mnist_config
from diffusion import DiffusionRunner

import os
#писали что не надо
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

config = create_default_mnist_config()
diffusion = DiffusionRunner(config)

diffusion.train()
