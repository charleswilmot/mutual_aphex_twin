import tensorflow as tf


class NetMaker:
    def __init__(self, network_dim):
        pass

    def __call__(self, inp):
        pass


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

iterators = {"A": some_tf_database_iterator,
             "B": some_tf_database_iterator}

train_tensors = {"networks": networks,
                 "losses": losses,
                 "train_op": train_ops,
                 "iterators": iterators}
