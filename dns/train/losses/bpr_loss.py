import tensorflow as tf


class BprLoss(tf.keras.layers.Layer):
    def __init__(self, movie_id_embedding: tf.keras.layers.Embedding):
        super().__init__()
        self._movie_id_embedding = movie_id_embedding

    def call(self, label, inputs, training):
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
        loss = tf.reduce_mean(tf.reduce_mean(loss, axis=1))
        return logits, loss


if __name__ == "__main__":
    label_logits = tf.constant([0, 1])
    logits = tf.constant([[1, 2, 3], [4, 5, 6]])
    # print(tf.squeeze(label_logits))
    print(tf.gather(logits, label_logits, batch_dims=1))
