import procedure
import network
import dataio


dim_A = 392
dim_B = 392
dim_latent_AA = 10
dim_latent_BA = 5
dim_latent_AB = 5
dim_latent_BB = 10
mode = "late_BN"
nlayers = 3

iterator = dataio.get_iterator("../data_in/left_right_mnist.tfr", buffer_size=2000, batch_size=512)
network_makers = network.get_network_makers(
    dim_A, dim_B, dim_latent_AA, dim_latent_BA, dim_latent_AB, dim_latent_BB, mode, nlayers)
tensors = network.get_tensors(network_makers, iterator)

procedure.run(tensors, None, None)
