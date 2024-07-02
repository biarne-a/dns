import tensorflow as tf


class MaxBprLoss(tf.keras.layers.Layer):
    def __init__(
        self,
        movie_id_embedding: tf.keras.layers.Embedding
    ):
        super().__init__()
        self._movie_id_embedding = movie_id_embedding

    def call(self, label, inputs, training):
        logits = tf.matmul(inputs, tf.transpose(self._movie_id_embedding.embeddings))
        label = tf.squeeze(label)
        negative_logits_mask = 1 - tf.one_hot(label, logits.shape[1])
        negative_logits = tf.multiply(logits, negative_logits_mask)

        # max_neg_logits_indices = tf.argmax(negative_logits, axis=1)
        # max_neg_logits = tf.gather(logits, max_neg_logits_indices, batch_dims=1)
        top_neg_logits = tf.math.top_k(negative_logits, k=100).values

        # Compute the difference in logits from positive labels and negative labels
        positive_logits = tf.gather(logits, label, batch_dims=1)
        diff = tf.transpose(positive_logits - tf.transpose(top_neg_logits))

        # Compute BPR loss
        diff_prob = tf.nn.sigmoid(diff)

        loss = -tf.reduce_sum(tf.math.log(diff_prob), axis=1)
        # loss = -tf.math.log(diff_prob)

        loss = tf.reduce_mean(loss)
        return logits, loss


if __name__ == "__main__":
    label_logits = tf.constant([0, 1])
    logits = tf.constant([[1, 2, 3], [4, 5, 6]])
    # print(tf.squeeze(label_logits))
    print(tf.gather(logits, label_logits, batch_dims=1))
