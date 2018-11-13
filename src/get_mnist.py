import tensorflow as tf
import os
import gzip
import numpy
import urllib.request
import dataio

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = '../data_in/'
IMAGE_SIZE = 28
PIXEL_DEPTH = 255


def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5]."""
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (2 * data / PIXEL_DEPTH) - 1
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)
        return data


if __name__ == "__main__":
    data = extract_data(maybe_download("train-images-idx3-ubyte.gz"), 60000)
    dataio.to_tfrerord(data[:, :, :14], data[:, :, 14:], WORK_DIRECTORY + "left_right_mnist.tfr")
    dataio.to_tfrerord(data[:, :14, :], data[:, 14:, :], WORK_DIRECTORY + "top_bottom_mnist.tfr")
    dataio.to_tfrerord(data[:, :, 0::2], data[:, :, 1::2], WORK_DIRECTORY + "left_right_interlaced_mnist.tfr")
    dataio.to_tfrerord(data[:, 0::2, :], data[:, 1::2, :], WORK_DIRECTORY + "top_bottom_interlaced_mnist.tfr")
    dataio.to_tfrerord(data, data, WORK_DIRECTORY + "mnist.tfr")
