import re
from typing import Dict

import numpy as np
import tensorflow as tf

from dns.train.config import Config


class Data:
    def __init__(
        self,
        train_ds: tf.data.Dataset,
        nb_train: int,
        test_ds: tf.data.Dataset,
        nb_test: int,
        movie_id_counts: Dict[str, int],
    ):
        self.train_ds = train_ds
        self.nb_train = nb_train
        self.test_ds = test_ds
        self.nb_test = nb_test
        self.movie_id_counts = movie_id_counts

    @property
    def movie_id_vocab(self):
        return list(self.movie_id_counts.keys())


def _read_unique_train_movie_id_counts(bucket_dir):
    with tf.io.gfile.GFile(f"{bucket_dir}/vocab/train_movie_counts.txt-00000-of-00001") as f:
        unique_train_movie_id_counts = {}
        for line in f.readlines():
            match = re.match("^\(([0-9]+), ([0-9]+)\)$", line.strip())  # noqa: W605
            movie_id = match.groups()[0]
            count = int(match.groups()[1])
            unique_train_movie_id_counts[movie_id] = count
    return unique_train_movie_id_counts


def get_data(config: Config):
    unique_train_movie_id_counts = _read_unique_train_movie_id_counts(config.data_dir)

    train_filenames = f"{config.data_dir}/tfrecords/train/*.gz"
    train = tf.data.Dataset.list_files(train_filenames, seed=42)
    train = train.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        cycle_length=8,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    test_filenames = f"{config.data_dir}/tfrecords/test/*.gz"
    test = tf.data.Dataset.list_files(test_filenames, seed=42)
    test = test.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        cycle_length=8,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    feature_description = {
        "context_movie_id": tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(0, 10)),
        "label_movie_id": tf.io.FixedLenFeature([1], tf.int64, default_value=0),
    }
    movie_id_vocab = list(unique_train_movie_id_counts.keys())
    movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=movie_id_vocab)

    def _parse_function(example_proto):
        x = tf.io.parse_single_example(example_proto, feature_description)
        return {
            "context_movie_id": movie_id_lookup(tf.strings.as_string(x["context_movie_id"])),
            "label_movie_id": movie_id_lookup(tf.strings.as_string(x["label_movie_id"])),
        }

    train_ds = train.map(_parse_function).repeat().batch(config.batch_size)

    test_ds = test.map(_parse_function).repeat().batch(config.batch_size)

    nb_train = 20278780
    nb_test = 2982077

    return Data(train_ds, nb_train, test_ds, nb_test, unique_train_movie_id_counts)
