import tensorflow as tf
from typing import Dict

from tensorflow.python.ops.nn_impl import _compute_sampled_logits

from dns.train.losses.bpr_loss import BprLoss


class SampledBprLoss(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_length: int,
        movie_id_counts: Dict[str, int],
        movie_id_embedding: tf.keras.layers.Embedding
    ):
        super().__init__()
        self._vocab_length = vocab_length
        self._movie_id_embedding = movie_id_embedding
        self._movie_id_biases = tf.zeros(shape=[self._vocab_length + 1], dtype=tf.float32)
        self._label_modalities_proba = list(movie_id_counts.values())
        self._test_loss = BprLoss(movie_id_embedding)

    def call(self, label, inputs, training):
        if not training:
            return self._test_loss(label, inputs, training)

        # num_neg_samples = self._vocab_length
        num_neg_samples = 100
        # num_neg_samples = self._vocab_length // 45

        sample_range = self._vocab_length + 1  # +1 to account for the default embedding at index 0
        labels = tf.reshape(tf.cast(label, dtype=tf.int64), [-1, 1])
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

        logits, label = _compute_sampled_logits(
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

        # Compute the difference in logits from positive labels and negative labels
        label_logits = logits[:, 0]
        diff = tf.transpose(label_logits - tf.transpose(logits))
        # Exclude differences from positive labels with itself
        diff = diff[:, 1:]
        # Compute BPR loss
        loss = -tf.math.log(tf.nn.sigmoid(diff), axis=1)
        loss = tf.reduce_mean(tf.reduce_mean(loss, axis=1))
        return logits, loss


if __name__ == "__main__":
    label_logits = tf.constant([0, 1])
    logits = tf.constant([[1, 2, 3], [4, 5, 6]])
    # print(tf.squeeze(label_logits))
    print(tf.gather(logits, label_logits, batch_dims=1))
