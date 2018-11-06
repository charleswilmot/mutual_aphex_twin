import tensorflow as tf


class NetMaker:
    def __init__(self, network_dim):
        self.network_dim = network_dim
        self._define_variables()

    def _define_variables(self):
        self.weights = []
        self.biases = []
        for in_dim, out_dim in zip(self.network_dim, self.network_dim[1:]):
            # TODO
            self.weights.append(tf.Variables(???))
            self.biases.append(tf.Variables(???))
        self.variables = self.weights + self.biases

    def _define_layer(self, prev, W, B):
        return tf.nn.relu(tf.matmul(prev, W) + B)

    def __call__(self, inp):
        prev = inp
        for var in zip(self.weights, self.biases):
            prev = self._define_layer(prev, *var)
        return prev


networks = {"AA": {"latent": some_placeholder_with_default,
                   "latent_statistics": some_moving_mean_stddev_of_the_latent_vect,
                   "out": network_output_tensor},
            "BA": {"latent": some_placeholder_with_default,
                   "latent_statistics": some_moving_mean_stddev_of_the_latent_vect,
                   "out": network_output_tensor},
            "AB": {"latent": some_placeholder_with_default,
                   "latent_statistics": some_moving_mean_stddev_of_the_latent_vect,
                   "out": network_output_tensor},
            "BB": {"latent": some_placeholder_with_default,
                   "latent_statistics": some_moving_mean_stddev_of_the_latent_vect,
                   "out": network_output_tensor}}

losses = {"AA": None,
          "BA": None,
          "AB": None,
          "BB": None,
          "equal": AB_latent_equals_BA_latent_mse}

train_op = tf.train.SomeOptimizer(lr).minimize(sum_of_losses)

iterator = some_tf_database_iterator

tensors = {"networks": networks,
           "losses": losses,
           "train_op": train_ops,
           "iterator": iterator}


network_makers = {"AA": {"inp_latent": some_netmaker, "latent_out": some_other_netmaker},
                  "BA": {"inp_latent": some_netmaker, "latent_out": some_other_netmaker},
                  "AB": {"inp_latent": some_netmaker, "latent_out": some_other_netmaker},
                  "BB": {"inp_latent": some_netmaker, "latent_out": some_other_netmaker}}


def automatic_layer_dim(start_dim, end_dim, nlayers, mode):
    if mode == "early_BN":
        return [start_dim] + [end_dim] * nlayers if start_dim > end_dim else [start_dim] * nlayers + [end_dim]
    if mode == "late_BN":
        return [start_dim] + [end_dim] * nlayers if start_dim < end_dim else [start_dim] * nlayers + [end_dim]
    if mode == "linear":
        return list(range(start_dim, end_dim, int((-start_dim + end_dim) // nlayers)))[:nlayers] + [end_dim]


def get_network_makers(dim_A, dim_B, dim_latent_AA, dim_latent_BA, dim_latent_AB, dim_latent_BB, mode, nlayers):
    network_makers = {}  # see line 57
    # use the "automatic_layer_dim" function
    return network_makers


def get_iterator(???):
    # already implemented in dataio.py
    pass


def get_networks(network_makers, iterator):
    networks = {}
    return networks


def get_losses(networks):
    losses = {}
    return losses


def get_train_op(losses):
    return train_op


def get_tensors(network_makers, iterator):
    networks = get_networks(network_makers, iterator)
    losses = get_losses(networks)
    train_op = get_train_op(losses)
    tensors = {"networks": networks,
               "losses": losses,
               "train_op": train_ops,
               "iterator": iterator}
    return tensors
