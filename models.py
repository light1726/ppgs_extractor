import tensorflow as tf
from tensorflow import keras


class DNNClassifier(object):
    def __init__(self, out_dims, hiddens, name):
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.dense_layers = []
            for i, units in enumerate(hiddens):
                dense = keras.layers.Dense(units, name='dense{}'.format(i))
                self.dense_layers.append(dense)
            self.output_dense = keras.layers.Dense(units=out_dims)

    def __call__(self, inputs, labels=None, lengths=None):
        """
        :param inputs: [batch, time, in_dims]
        :param labels: [batch, time, out_dims], should be one-hot
        :param lengths: [batch]
        :return:
        """
        cur_layer = inputs
        for layer in self.dense_layers:
            cur_layer = layer(cur_layer)
        logits = self.output_dense(cur_layer)  # [batch, time, out_dims]
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(labels),
            logits=logits,
            axis=-1) if labels is not None else None
        return {'logits': logits,  # [time, dims]
                'cross_entropy': cross_entropy}
