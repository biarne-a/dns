import tensorflow as tf

from dns.train.losses.bpr_loss import BprLoss


class SampledBprLoss(tf.keras.layers.Layer):
    def __init__(self, movie_id_embedding: tf.keras.layers.Embedding):
        super().__init__()
        self._movie_id_embedding = movie_id_embedding
        self._test_loss = BprLoss(movie_id_embedding)

    def call(self, label, inputs, training):
        if not training:
            return self._test_loss(label, inputs, training)

        logits = tf.matmul(inputs, tf.transpose(self._movie_id_embedding.embeddings))
        # Compute the difference in logits from positive labels and negative labels
        label = tf.squeeze(label)
        label_logits = tf.gather(logits, label, batch_dims=1)
        diff = tf.transpose(label_logits - tf.transpose(logits))
        # Compute BPR loss
        loss = -tf.math.log(tf.nn.sigmoid(diff))
        # Exclude differences from positive labels with itself
        label_mask = 1 - tf.one_hot(label, logits.shape[1])
        loss = tf.multiply(loss, label_mask)

        loss = tf.reduce_mean(loss)
        return logits, loss


if __name__ == "__main__":
    label_logits = tf.constant([0, 1])
    logits = tf.constant([[1, 2, 3], [4, 5, 6]])
    # print(tf.squeeze(label_logits))
    print(tf.gather(logits, label_logits, batch_dims=1))
