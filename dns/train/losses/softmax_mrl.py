import keras
import tensorflow as tf


class SoftmaxMRL(keras.layers.Layer):
    def __init__(self, movie_id_embedding: keras.layers.Embedding):
        super().__init__()
        self._movie_id_embedding = movie_id_embedding
        self._loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def _get_level_loss(self, label, inputs, max_level_dim):
        level_inputs = inputs[:, :max_level_dim]
        movie_id_embedding = self._movie_id_embedding.embeddings[:, :max_level_dim]

        logits = tf.matmul(level_inputs, tf.transpose(movie_id_embedding))
        probs = tf.nn.softmax(logits)
        loss_value = self._loss(label, probs)
        return logits, loss_value

    def call(self, label, inputs, training):
        _, loss_16 = self._get_level_loss(label, inputs, 16)
        _, loss_32 = self._get_level_loss(label, inputs, 32)
        _, loss_64 = self._get_level_loss(label, inputs, 64)
        logits, loss_128 = self._get_level_loss(label, inputs, 128)
        loss = tf.reduce_mean(loss_16 + loss_32 + loss_64 + loss_128)

        return logits, loss
