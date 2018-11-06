import tensorflow as tf
from collecitons import defaultdict


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


def automatic_layer_dim(start_dim, end_dim, nlayers, mode):
    if mode == "early_BN":
        return [start_dim] + [end_dim] * nlayers if start_dim > end_dim else [start_dim] * nlayers + [end_dim]
    if mode == "late_BN":
        return [start_dim] + [end_dim] * nlayers if start_dim < end_dim else [start_dim] * nlayers + [end_dim]
    if mode == "linear":
        return list(range(start_dim, end_dim, int((-start_dim + end_dim) // nlayers)))[:nlayers] + [end_dim]


def get_network_makers(dim_A, dim_B, dim_latent_AA, dim_latent_BA, dim_latent_AB, dim_latent_BB, mode, nlayers):
    network_makers = defaultdict()  # see line 57

    network_maker["AA"]["inp_latent"] = NetMaker(automatic_layer_dim(dim_A, dim_latent_AA, nlayers, mode))
    network_maker["AA"]["latent_out"] = NetMaker(automatic_layer_dim(dim_latent_AA, dim_A, nlayers, mode))

    network_maker["BB"]["inp_latent"] = NetMaker(automatic_layer_dim(dim_B, dim_latent_BB, nlayers, mode))
    network_maker["BB"]["latent_out"] = NetMaker(automatic_layer_dim(dim_latent_BB, dim_B, nlayers, mode))

    network_maker["AB"]["imp_latent"] = NetMaker(automatic_layer_dim(dim_A, dim_latent_AB, nlayers, mode))
    network_maker["AB"]["latent_out"] = NetMaker(automatic_layer_dim(dim_latent_AB, dim_B, nlayers, mode))

    network_maker["BA"]["imp_latent"] = NetMaker(automatic_layer_dim(dim_B, dim_latent_BA, nlayers, mode))
    network_maker["BA"]["latent_out"] = NetMaker(automatic_layer_dim(dim_latent_BA, dim_A, nlayers, mode))

    return network_makers


def get_networks(network_makers, iterator):
    A, B = iterator.get_next()
    for key in network_makers:
        inp = A if key[0] == 'A' else B
        latent = network_makers[key]["inp_latent"](inp)
        networks[key] = {}
        networks[key]["latent"] = tf.placeholder_with_default(latent, shape=latent.shape.as_list())
        if key in ["AB", "BA"]:
            networks[key]["out"] = network_makers[key]["latent_out"](networks[key]["latent"])
    con = tf.concat([networks["AA"]["latent"], networks["BA"]["latent"]], axis=0)
    networks["AA"]["out"] = network_makers["AA"]["latent_out"](con)
    con = tf.concat([networks["BB"]["latent"], networks["AB"]["latent"]], axis=0)
    networks["BB"]["out"] = network_makers["BB"]["latent_out"](con)
    return networks


def mse(a, b):
    return tf.reduce_mean((a - b) * (a - b))


def get_losses(networks, iterator):
    A, B = iterator.get_next()
    losses = {"AA": mse(A, networks["AA"]["out"]),
              "BA": mse(A, networks["BA"]["out"]),
              "AB": mse(B, networks["AB"]["out"]),
              "BB": mse(B, networks["BB"]["out"]),
              "equal": mse(networks["AB"]["latent"], networks["BA"]["latent"])}
    return losses


def get_train_op(losses):
    sum_of_losses = losses["AA"] + losses["BB"] + losses["AB"] + losses["BA"] + losses["equal"]
    train_op = tf.train.AdamOptimizer().minimize(sum_of_losses)
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
