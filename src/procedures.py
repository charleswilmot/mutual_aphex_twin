import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


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


def get_data_points(sess, inputs, n):
    np_As, np_Bs = sess.run(inputs)
    return np_As[:n], np_Bs[:n]


def get_noises(stats, ninterpolations, nstddev):
    noises = {}
    for key in stats:
        shape = (ninterpolations, stats[key]["mean"].shape[0])
        noises[key] = stats[key]["mean"] + nstddev * stats[key]["std"] * np.random.normal(size=shape)
    return noises


def test(sess, networks, losses, inputs, ninterpolations, nstddev, nbatches):
    ### Compute losses on nbatches and retrieve statistics of latent spaces
    latents = {
        "AA": networks["AA"]["latent"],
        "AB": networks["AB"]["latent"],
        "BA": networks["BA"]["latent"],
        "BB": networks["BB"]["latent"]
    }
    np_latents = {
        "AA": [],
        "AB": [],
        "BA": [],
        "BB": []
    }
    np_losses = {
        "AA": [],
        "AB": [],
        "BA": [],
        "BB": [],
        "equal": []
    }
    for i in range(nbatches):
        np_losses_tmp, np_latents_tmp = sess.run([losses, latents])
        for key in np_losses:
            np_losses[key].append(np_losses_tmp[key])
        for key in np_latents:
            np_latents[key].append(np_latents_tmp[key])
    for key in np_losses:
        np_losses[key] = np.mean(np_losses_tmp[key])
    stats = {}
    for key in np_latents:
        tmp = np.vstack(np_latents_tmp[key])
        stats[key] = {"mean": np.mean(tmp, axis=0), "std": np.std(tmp, axis=0)}
    ### Get n data points
    np_As, np_Bs = get_data_points(sess, inputs, n=ninterpolations)
    ### Generate n noise for each latent spaces according to stats
    np_noises = get_noises(stats, ninterpolations=ninterpolations, nstddev=nstddev)
    ### Compute reconstructions
    reconstructions = {}
    ### compute reconstructions from noise only
    fetches = {
        "AB": networks["AB"]["out"],
        "BA": networks["BA"]["out"],
        "AA": networks["AA"]["out"],
        "BB": networks["BB"]["out"]
    }
    feed_dict = {
        networks["AA"]["latent"]: np_noises["AA"],
        networks["AB"]["latent"]: np_noises["AB"],
        networks["BA"]["latent"]: np_noises["BA"],
        networks["BB"]["latent"]: np_noises["BB"]
    }
    reconstructions["noise"] = sess.run(fetches, feed_dict=feed_dict)
    ### compute reconstructions from A
    fetches = {
        "A": inputs[0],
        "AB": networks["AB"]["out"],
        "BA": networks["BA"]["out"],
        "AA": networks["AA"]["out"],
        "BB": networks["BB"]["out"]
    }
    feed_dict = {
        inputs[0]: np_As,
        networks["BA"]["latent"]: np_noises["BA"],
        networks["BB"]["latent"]: np_noises["BB"]
    }
    reconstructions["A"] = sess.run(fetches, feed_dict=feed_dict)
    ### compute reconstructions from B
    fetches = {
        "B": inputs[1],
        "AB": networks["AB"]["out"],
        "BA": networks["BA"]["out"],
        "AA": networks["AA"]["out"],
        "BB": networks["BB"]["out"]
    }
    feed_dict = {
        inputs[1]: np_Bs,
        networks["AA"]["latent"]: np_noises["AA"],
        networks["AB"]["latent"]: np_noises["AB"]
    }
    reconstructions["B"] = sess.run(fetches, feed_dict=feed_dict)
    ### compute reconstructions from A and B
    fetches = {
        "A": inputs[0],
        "B": inputs[1],
        "AB": networks["AB"]["out"],
        "BA": networks["BA"]["out"],
        "AA": networks["AA"]["out"],
        "BB": networks["BB"]["out"]
    }
    feed_dict = {
        inputs[0]: np_As,
        inputs[1]: np_Bs
    }
    reconstructions["AB"] = sess.run(fetches, feed_dict=feed_dict)
    ### return reconstruction and losses
    return reconstructions, np_losses


def mnist_to_uint8(X):
    return (((np.clip(X, -1, 1) + 1) * 255) / 2).astype(np.uint8)


def left_right_mnist_recreate(A, B):
    n = A.shape[0]
    C = np.zeros((n, 28, 28), dtype=np.uint8)
    C[:, :, :14] = mnist_to_uint8(A).reshape((n, 28, 14))
    C[:, :, 14:] = mnist_to_uint8(B).reshape((n, 28, 14))
    return C


def top_bottom_mnist_recreate(A, B):
    n = A.shape[0]
    C = np.zeros((n, 28, 28), dtype=np.uint8)
    C[:, :14, :] = mnist_to_uint8(A).reshape((n, 14, 28))
    C[:, 14:, :] = mnist_to_uint8(B).reshape((n, 14, 28))
    return C


def left_right_interlaced_mnist_recreate(A, B):
    n = A.shape[0]
    C = np.zeros((n, 28, 28), dtype=np.uint8)
    C[:, :, 0::2] = mnist_to_uint8(A).reshape((n, 28, 14))
    C[:, :, 1::2] = mnist_to_uint8(B).reshape((n, 28, 14))
    return C


def top_bottom_interlaced_mnist_recreate(A, B):
    n = A.shape[0]
    C = np.zeros((n, 28, 28), dtype=np.uint8)
    C[:, 0::2, :] = mnist_to_uint8(A).reshape((n, 14, 28))
    C[:, 1::2, :] = mnist_to_uint8(B).reshape((n, 14, 28))
    return C


def simple_loss_plot(ax, losses):
    for key in losses[0]:
        tmp = []
        for l in losses:
            tmp.append(l[key])
        ax.plot(tmp, label=key)
    ax.legend()


def recreate_reconstructions(reconstructions, recreate_function, default_A, default_B):
    keys = [
        "A", "B", "A(B)", "B(A)",
        "A(A,B)", "B(A,B)", "A+B", "A+B(A)",
        "B+A(B)", "A+B(A,B)", "A(A,B)+B", "A(A,B)+B(A,B)",
        "A(B)+B(A)", "A(A,B)+B(A)", "A(B)+B(A,B)"
    ]
    recs = reconstructions
    recreation = {}
    recreation["noise"] = {}
    for key in ["noise", "A", "B", "AB"]:
        A = recs[key]["A"] if "A" in recs[key] else default_A
        B = recs[key]["B"] if "B" in recs[key] else default_B
        recreation[key] = {}
        recreation[key]["A"] = recreate_function(A, default_B)
        recreation[key]["B"] = recreate_function(default_A, B)
        recreation[key]["A(B)"] = recreate_function(recs[key]["BA"], default_B)
        recreation[key]["B(A)"] = recreate_function(default_A, recs[key]["AB"])
        recreation[key]["A(A,B)"] = recreate_function(recs[key]["AA"], default_B)
        recreation[key]["B(A,B)"] = recreate_function(default_A, recs[key]["BB"])
        recreation[key]["A+B"] = recreate_function(A, B)
        recreation[key]["A+B(A)"] = recreate_function(A, recs[key]["AB"])
        recreation[key]["B+A(B)"] = recreate_function(recs[key]["BA"], B)
        recreation[key]["A+B(A,B)"] = recreate_function(A, recs[key]["BB"])
        recreation[key]["A(A,B)+B"] = recreate_function(recs[key]["AA"], B)
        recreation[key]["A(A,B)+B(A,B)"] = recreate_function(recs[key]["AA"], recs[key]["BB"])
        recreation[key]["A(B)+B(A)"] = recreate_function(recs[key]["BA"], recs[key]["AB"])
        recreation[key]["A(A,B)+B(A)"] = recreate_function(recs[key]["AA"], recs[key]["AB"])
        recreation[key]["A(B)+B(A,B)"] = recreate_function(recs[key]["BA"], recs[key]["BB"])
    return recreation


def plot_recreations_vertical(ax, recreations):
    example = recreations["noise"]["A"]
    HEADER_SIZE = 120
    BIG_MARGIN = 60
    LITTLE_MARGIN = 2
    N_SECTIONS = len(recreations)
    N_COLUMNS = len(recreations["noise"])
    N_PATCH_PER_SECTION = example.shape[0]
    PATCH_HEIGHT = example.shape[1]
    PATCH_WIDTH = example.shape[2]
    LINE_HEIGHT = LITTLE_MARGIN + PATCH_HEIGHT
    COLUMN_WIDTH = LITTLE_MARGIN + PATCH_WIDTH
    SECTION_SIZE = LINE_HEIGHT * N_PATCH_PER_SECTION + BIG_MARGIN
    IMAGE_HEIGHT = HEADER_SIZE + SECTION_SIZE * N_SECTIONS
    IMAGE_WIDTH = COLUMN_WIDTH * N_COLUMNS + LITTLE_MARGIN
    image = np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    column_names = ["A+B", "A", "B", "A(B)", "B(A)", "A(A,B)", "B(A,B)", "A+B(A)", "B+A(B)", "A+B(A,B)", "A(A,B)+B",
                    "A(A,B)+B(A,B)", "A(B)+B(A)", "A(A,B)+B(A)", "A(B)+B(A,B)"]
    for section_number, section_name in enumerate(["noise", "A", "B", "AB"]):
        section_height_start = HEADER_SIZE + section_number * SECTION_SIZE
        section_width_start = LITTLE_MARGIN
        for column_number, column_name in enumerate(column_names):
            column_width_start = section_width_start + column_number * COLUMN_WIDTH
            ax.text(column_width_start, HEADER_SIZE / 2, column_name, fontsize=4)
            for line_number, patch in enumerate(recreations[section_name][column_name]):
                line_height_start = section_height_start + line_number * LINE_HEIGHT
                h = line_height_start
                w = column_width_start
                image[h:h + PATCH_HEIGHT, w:w + PATCH_WIDTH] = patch
    ax.imshow(image)


def plot_recreations_horizontal(ax, recreations):
    example = recreations["noise"]["A"]
    HEADER_SIZE = 120
    BIG_MARGIN = 60
    LITTLE_MARGIN = 2
    N_SECTIONS = len(recreations)
    N_LINES = len(recreations["noise"])
    N_PATCH_PER_SECTION = example.shape[0]
    PATCH_HEIGHT = example.shape[1]
    PATCH_WIDTH = example.shape[2]
    LINE_HEIGHT = LITTLE_MARGIN + PATCH_HEIGHT
    COLUMN_WIDTH = LITTLE_MARGIN + PATCH_WIDTH
    SECTION_SIZE = COLUMN_WIDTH * N_PATCH_PER_SECTION + BIG_MARGIN
    IMAGE_WIDTH = HEADER_SIZE + SECTION_SIZE * N_SECTIONS
    IMAGE_HEIGHT = LINE_HEIGHT * N_LINES + LITTLE_MARGIN
    image = np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    line_names = ["A+B", "A", "B", "A(B)", "B(A)", "A(A,B)", "B(A,B)", "A+B(A)", "B+A(B)", "A+B(A,B)", "A(A,B)+B",
                  "A(A,B)+B(A,B)", "A(B)+B(A)", "A(A,B)+B(A)", "A(B)+B(A,B)"]
    for section_number, section_name in enumerate(["noise", "A", "B", "AB"]):
        section_width_start = HEADER_SIZE + section_number * SECTION_SIZE
        section_height_start = LITTLE_MARGIN
        for line_number, line_name in enumerate(line_names):
            line_height_start = section_height_start + line_number * LINE_HEIGHT
            ax.text(2, line_height_start + PATCH_HEIGHT // 2, line_name, fontsize=4)
            for column_number, patch in enumerate(recreations[section_name][line_name]):
                column_width_start = section_width_start + column_number * COLUMN_WIDTH
                h = line_height_start
                w = column_width_start
                image[h:h + PATCH_HEIGHT, w:w + PATCH_WIDTH] = patch
    ax.imshow(image)


def run(tensors, recreate_function, default_A, default_B):
    n_train_batches = 200
    n_test_batches = 20
    all_test_losses = []
    with tf.Session() as sess:
        sess.run(tensors["iterator"].initializer)
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            train_losses = train(sess, tensors["train_op"], tensors["losses"], n_train_batches)
            recs, test_losses = test(sess, tensors["networks"], tensors["losses"], tensors["inputs"], 6, 1, n_test_batches)
            all_test_losses.append(test_losses)
            print(test_losses)
            recreations = recreate_reconstructions(recs, recreate_function, default_A, default_B)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            plot_recreations_horizontal(ax, recreations)
            fig.savefig("../data_out/{}.png".format(i), dpi=300, bbox_inches='tight')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    simple_loss_plot(ax, all_test_losses)
    fig.savefig("../data_out/loss.png", dpi=300, bbox_inches='tight')
