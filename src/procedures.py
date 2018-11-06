import tensorflow as tf


def train(sess, train_op, losses, nbatches):
    # returns the mean of each loss averaged on nbatches
    if nbatches <= 0:
        raise ValueError("Can't compute {} batches (<=0)".format(nbatches))
    _, np_losses = sess.run([train_op, losses])
    for i in range(nbatches - 1):
        _, tmp_np_losses = sess.run([train_op, losses])
        for key in tmp_np_losses:
            np_losses[key] += tmp_np_losses[key]
    for key in np_losses:
        np_losses[key] /= nbatches
    return np_losses


def test(sess, networks, losses, iterator, ninterpolations, nstddev, nbatches):
    # returns the mean of each loss averaged on training set (as dict) and n reconstructions after latent space interpolation for each net (as dict) (plus original image used... complex I know)
    pass


def left_right_mnist_recreate(A, B):
    C = np.zeros((A.shape[0], 28, 28), dtype=np.uint8)
    C[:, :, :14] = (((A + 1) * 255) / 2).astype(np.uint8)
    C[:, :, 14:] = (((B + 1) * 255) / 2).astype(np.uint8)
    return C


def top_bottom_mnist_recreate(A, B):
    C = np.zeros((A.shape[0], 28, 28), dtype=np.uint8)
    C[:, :14, :] = (((A + 1) * 255) / 2).astype(np.uint8)
    C[:, 14:, :] = (((B + 1) * 255) / 2).astype(np.uint8)
    return C


def left_right_interlaced_mnist_recreate(A, B):
    C = np.zeros((A.shape[0], 28, 28), dtype=np.uint8)
    C[:, :, 0::2] = (((A + 1) * 255) / 2).astype(np.uint8)
    C[:, :, 1::2] = (((B + 1) * 255) / 2).astype(np.uint8)
    return C


def top_bottom_interlaced_mnist_recreate(A, B):
    C = np.zeros((A.shape[0], 28, 28), dtype=np.uint8)
    C[:, 0::2, :] = (((A + 1) * 255) / 2).astype(np.uint8)
    C[:, 1::2, :] = (((B + 1) * 255) / 2).astype(np.uint8)
    return C


def mnist_interpolation_plot(fig, recreated):
    ax = fig.add_subplot(321)
    plot_something(ax, recreated["something"], recreated["something_else"])
    ax = fig.add_subplot(322)
    plot_something_else(ax, recreated["something"], recreated["something_else"])


def split_mnist_interpolation_plot(fig, reconstructions):
    recreated = split_mnist_recreate(reconstructions["something_A"], reconstructions["something_B"])
    mnist_interpolation_plot(fig, recreated)


def interlaced_mnist_interpolation_plot(fig, reconstructions):
    recreated = interlaced_mnist_recreate(reconstructions["something_A"], reconstructions["something_B"])
    mnist_interpolation_plot(fig, recreated)


def simple_loss_plot(ax):
    pass


def run(tensors, losses_plot_function, interpolation_plot_function):
    nbatches = 100
    with tf.Session() as sess:
        sess.run(tensors["iterator"].initializer)
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            losses = train(sess, tensors["train_op"], tensors["losses"], nbatches)
            print(losses)
