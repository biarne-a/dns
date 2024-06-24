import keras
import tensorflow as tf


class SoftmaxCrossEntropy(keras.layers.Layer):
    def __init__(self, movie_id_embedding: keras.layers.Embedding):
        super().__init__()
        self._movie_id_embedding = movie_id_embedding
        self._loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def call(self, label, inputs, training):
        logits = tf.matmul(inputs, tf.transpose(self._movie_id_embedding.embeddings))
        probs = tf.nn.softmax(logits)
        loss_value = self._loss(label, probs)
        return logits, loss_value
