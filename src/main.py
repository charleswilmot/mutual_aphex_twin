import procedures
import network
import dataio
import numpy as np


dim_A = 392
dim_B = 392
dim_latent_AA = 2
dim_latent_BA = 2
dim_latent_AB = 2
dim_latent_BB = 2
mode = "linear"
nlayers = 3

iterator = dataio.get_iterator("../data_in/top_bottom_interlaced_mnist.tfr", buffer_size=2000, batch_size=512)
network_makers = network.get_network_makers(
    dim_A, dim_B, dim_latent_AA, dim_latent_BA, dim_latent_AB, dim_latent_BB, mode, nlayers)
tensors = network.get_tensors(network_makers, iterator)

default = np.full((6, 28 * 14), -1, dtype=np.float32)
procedures.run(tensors, procedures.top_bottom_interlaced_mnist_recreate, default, default)
