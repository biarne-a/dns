from typing import Dict

import keras
import tensorflow as tf
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from tensorflow.python.ops import nn_ops

from dns.train.losses.softmax_xe import SoftmaxCrossEntropy


class SampledSoftmaxCrossEntropy(keras.layers.Layer):
    def __init__(self, vocab_length: int, movie_id_counts: Dict[str, int], movie_id_embedding: keras.layers.Embedding):
        super().__init__()
        self._vocab_length = vocab_length
        self._movie_id_embedding = movie_id_embedding
        self._movie_id_biases = tf.zeros(shape=[self._vocab_length + 1], dtype=tf.float32)
        # self._label_modalities_proba = [1 / (self._vocab_length + 1)] * self._vocab_length
        self._label_modalities_proba = list(movie_id_counts.values())
        self._test_loss = SoftmaxCrossEntropy(movie_id_embedding)

    def call(self, label, inputs, training):
        if not training:
            return self._test_loss(label, inputs, training)

        sample_range = self._vocab_length + 1   # +1 to account for the default embedding at index 0
        labels = tf.reshape(tf.cast(label, dtype=tf.int64), [-1, 1])
        # num_neg_samples = self._vocab_length // 45
        num_neg_samples = 100
        sampled_values = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,  # list of target-ids (ground-truths) [batch_size x _nb_videos]
            num_true=1,  # ground-truth labels are vectors of len 1 (not a multi-class classification)
            num_sampled=num_neg_samples,  # how many samples to extract for loss computation -> impacts the exec time
            unique=True,  # do not sample the same index/video_id twice for the same batch
            range_max=sample_range,  # number of distinct classes = video vocab
            unigrams=self._label_modalities_proba,
            # list of video occurrences in dataset = sampling probabilites  # noqa:501
            distortion=0.4,  # how much to flatten the unigrams distribution (1.0=unchanged, 0.0=uniform sampling)
            num_reserved_ids=1,  # adds a sampling proba of 0.0 at index 0 to exclude default embedding
            seed=42,
        )

        sampled_logits, sampled_labels = _compute_sampled_logits(
            weights=self._movie_id_embedding.embeddings,
            biases=self._movie_id_biases,
            labels=labels,
            inputs=inputs,
            num_sampled=num_neg_samples,
            num_classes=self._vocab_length,
            sampled_values=sampled_values,
            subtract_log_q=True,
            seed=42,
        )
        sampled_labels = tf.stop_gradient(sampled_labels)

        sampled_losses = nn_ops.softmax_cross_entropy_with_logits_v2(
            labels=sampled_labels, logits=sampled_logits)
        loss = tf.reduce_mean(sampled_losses)

        return sampled_logits, loss
