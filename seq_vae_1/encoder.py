import tensorflow as tf
import tensorflow_probability as tfp

import common.lstm_utils as lstm_utils

rnn = tf.contrib.rnn
ds = tfp.distributions


class BidirectionalLstmEncoder(object):

    def __init__(self, hparams, name_or_scope='encoder'):
        self.name_or_scope = name_or_scope
        self._batch_size = hparams.batch_size
        self._seq_len = hparams.seq_len
        self._z_size = hparams.z_size
        self._free_bits = hparams.free_bits
        self._vocab_size = hparams.vocab_size
        self._emb_dim = hparams.emb_dim
        self._learning_rate = hparams.encoder_learning_rate
        self._grad_clip = 5.0

        cells_fw = []
        cells_bw = []
        for i, layer_size in enumerate(hparams.enc_rnn_size):
            cells_fw.append(
                lstm_utils.rnn_cell(
                    [layer_size],
                    hparams.dropout_keep_prob,
                    hparams.residual_encoder))
            cells_bw.append(
                lstm_utils.rnn_cell(
                    [layer_size],
                    hparams.dropout_keep_prob,
                    hparams.residual_encoder))

        self._cells_fw = cells_fw
        self._cells_bw = cells_bw

        self._embeddings = tf.Variable(tf.random_normal([self._vocab_size, self._emb_dim], stddev=0.1))

        self.seq = tf.placeholder(tf.int32, shape=[self._batch_size, self._seq_len])
        # self.seq_len = tf.placeholder(tf.int32, shape=[self._batch_size])
        self.seq_len = [self._seq_len] * self._batch_size

        seq_emb = tf.nn.embedding_lookup(self._embeddings, self.seq)

        with tf.variable_scope(self.name_or_scope, reuse=tf.AUTO_REUSE):
            _, states_fw, states_bw = rnn.stack_bidirectional_dynamic_rnn(
                self._cells_fw,
                self._cells_bw,
                seq_emb,
                sequence_length=self.seq_len,
                time_major=False,
                dtype=tf.float32, )

            last_c_fw = states_fw[-1][-1].h
            last_c_bw = states_bw[-1][-1].h
            output = tf.concat([last_c_fw, last_c_bw], 1)

            self.mu = tf.layers.dense(
                output,
                self._z_size,
                name='mu',
                kernel_initializer=tf.random_normal_initializer(stddev=0.001))
            self.sigma = tf.layers.dense(
                output,
                self._z_size,
                activation=tf.nn.softplus,
                name='sigma',
                kernel_initializer=tf.random_normal_initializer(stddev=0.001))

            self.z_q = ds.MultivariateNormalDiag(loc=self.mu, scale_diag=self.sigma)

            self.z_p = ds.MultivariateNormalDiag(
                loc=[0.] * self._z_size,
                scale_diag=[1.] * self._z_size)

            # compute formula:https://zhuanlan.zhihu.com/p/22464760
            self._kl_div = ds.kl_divergence(self.z_q, self.z_p)
            self.z_q_s = self.z_q.sample()
            free_nats = self._free_bits * tf.math.log(2.0)
            self.kl_loss = tf.reduce_mean(tf.maximum(self._kl_div - free_nats, 0))

            self.e_params = [param for param in tf.trainable_variables() if param.name.startswith(self.name_or_scope)]

            e_opt = self.e_optimizer(self._learning_rate)
            self.e_grad, _ = tf.clip_by_global_norm(tf.gradients(self.kl_loss, self.e_params), self._grad_clip)
            self.e_updates = e_opt.apply_gradients(zip(self.e_grad, self.e_params))

    def encoder(self, sess, seq):
        return sess.run(self.z_q_s, feed_dict={self.seq: seq})

    def train_step(self, sess, seq):
        z, _, kl_loss = sess.run([self.z_q_s, self.e_updates, self.kl_loss], feed_dict={self.seq: seq})
        return z, kl_loss

    def e_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)
