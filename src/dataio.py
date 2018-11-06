import tensorflow as tf
import numpy as np


def serialize(A, B):
    feature = {
        'A': tf.train.Feature(bytes_list=tf.train.BytesList(value=[A.astype(np.float32).tobytes()])),
        'B': tf.train.Feature(bytes_list=tf.train.BytesList(value=[B.astype(np.float32).tobytes()]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def to_tfrerord(As, Bs, output_file):
    with tf.python_io.TFRecordWriter(output_file) as writer:
        for A, B in zip(As, Bs):
            writer.write(serialize(A, B))


def parse_proto(example_proto):
    features = {
        'A': tf.FixedLenFeature([], tf.string),
        'B': tf.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return tf.decode_raw(parsed_features['A'], tf.float32), tf.decode_raw(parsed_features['B'], tf.float32)


def get_iterator(file_names, buffer_size=10000, batch_size=100):
    dataset = tf.data.TFRecordDataset(file_names)
    dataset = dataset.map(parse_proto)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset.make_initializable_iterator()


if __name__ == '__main__':
    import numpy as np

    output_file = "/tmp/test111.tfr"
    As = np.random.uniform(size=(1000, 2))
    Bs = np.random.uniform(size=(1000, 2))
    to_tfrerord(As, Bs, output_file)
    iterator = read_tfrecords(file_names=(output_file), buffer_size=200, batch_size=3)
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        print(sess.run(iterator.get_next()), "\n\n")
        print(sess.run(iterator.get_next()), "\n\n")
        print(sess.run(iterator.get_next()), "\n\n")
        print(sess.run(iterator.get_next()), "\n\n")
        print(sess.run(iterator.get_next()), "\n\n")
