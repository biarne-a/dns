import tensorflow as tf


class DynamicBprLoss(tf.keras.layers.Layer):
    def __init__(
        self,
        movie_id_embedding: tf.keras.layers.Embedding
    ):
        super().__init__()
        self._movie_id_embedding = movie_id_embedding

    def _get_level_loss(self, label, inputs, max_level_dim):
        level_inputs = inputs[:, :max_level_dim]
        movie_id_embedding = self._movie_id_embedding.embeddings[:, :max_level_dim]

        logits = tf.matmul(level_inputs, tf.transpose(movie_id_embedding))
        probs = tf.nn.softmax(logits)

        # Compute the difference in logits from positive labels and negative labels
        label = tf.squeeze(label)
        label_logits = tf.gather(logits, label, batch_dims=1)
        diff = tf.transpose(label_logits - tf.transpose(logits))

        # Compute BPR loss
        sup_max_prob = probs * tf.nn.sigmoid(diff)
        # Exclude differences from positive labels with itself
        label_mask = 1 - tf.one_hot(label, logits.shape[1])
        sup_max_prob = tf.multiply(sup_max_prob, label_mask)

        loss = -tf.math.log(tf.reduce_sum(sup_max_prob, axis=1))

        loss = tf.reduce_sum(loss)
        return logits, loss

    def call(self, label, inputs, training):
        _, loss_16 = self._get_level_loss(label, inputs, 16)
        _, loss_32 = self._get_level_loss(label, inputs, 32)
        _, loss_64 = self._get_level_loss(label, inputs, 64)
        logits, loss_128 = self._get_level_loss(label, inputs, 128)
        loss = tf.reduce_mean(loss_16 + loss_32 + loss_64 + loss_128)

        return logits, loss

if __name__ == "__main__":
    label_logits = tf.constant([0, 1])
    logits = tf.constant([[1, 2, 3], [4, 5, 6]])
    # print(tf.squeeze(label_logits))
    print(tf.gather(logits, label_logits, batch_dims=1))
