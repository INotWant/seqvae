import os
import sys

sys.path.append('../')

import pickle
import time

import numpy as np
import tensorflow as tf

import configs
from basic_vae_1.encoder import BidirectionalLstmEncoder
from basic_vae_1.generator import Generator
from seqgan.dataloader import Gen_Data_loader
from target_lstm import TARGET_LSTM

# GPU's configuration
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True


# only use CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_z(batch_size, z_size):
    return np.random.multivariate_normal([0.] * z_size, np.eye(z_size), batch_size)


def generate_samples(sess, generator, batch_size, generated_num, z_size, output_file):
    z = get_z(batch_size, z_size)
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(generator.generate(sess, z))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def train_epoch(sess, encoder, generator, data_loader, train_updates):
    # train the generator using MLE for one epoch
    supervised_kl_losses = []
    supervised_r_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, kl_loss, r_loss = sess.run(
            [train_updates, encoder.kl_loss, generator.train_loss],
            feed_dict={encoder.seq: batch, generator.x: batch}
        )
        supervised_kl_losses.append(kl_loss)
        supervised_r_losses.append(r_loss)
    return np.mean(supervised_kl_losses), np.mean(supervised_r_losses)


def get_train_updates(encoder, generator, learning_rate, grad_clip):
    loss = encoder.kl_loss + generator.train_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_grad, _ = tf.clip_by_global_norm(tf.gradients(loss, generator.params), grad_clip)
    return optimizer.apply_gradients(zip(train_grad, generator.params))


def main():
    time_str = str(int(time.time()))
    # configuration
    config = configs.CONFIG_MAP['t_8']
    target_params_file = '../' + config.target_params_file
    start_token = config.start_token
    assert start_token == 0
    generated_num = config.generated_num
    hparams = config.hparams

    TOTAL_EPOCH = 400
    grad_clip = 5.0
    batch_size = hparams.batch_size
    seq_len = hparams.seq_len
    z_size = hparams.z_size
    positive_file = '../' + config.positive_file_txt
    eval_file = '../save/eval_file.txt'

    gen_data_loader = Gen_Data_loader(batch_size, seq_len)
    likelihood_data_loader = Gen_Data_loader(batch_size, seq_len)  # For testing

    # log file
    hparams_str = hparams.__str__()
    log_fn = '../save/basic_vae_1_' + time_str + '.log'
    log_file = open(log_fn, 'a+')
    log_file.write(hparams_str + "\n")
    log_file.flush()

    graph = tf.get_default_graph()
    with graph.as_default():
        sess = tf.Session()

        target_params = pickle.load(open(target_params_file, 'rb'))
        target_lstm = TARGET_LSTM(
            hparams.vocab_size,
            batch_size,
            hparams.emb_dim,
            hparams.hidden_dim,
            hparams.seq_len,
            start_token,
            target_params)  # The oracle model

        encoder = BidirectionalLstmEncoder(hparams)
        generator = Generator(hparams, start_token, encoder)

        train_updates = get_train_updates(encoder, generator, hparams.dencoder_learning_rate, grad_clip)

        gen_data_loader.create_batches(positive_file)
        sess.run(tf.global_variables_initializer())

        print('#########################################################################', file=log_file)
        print('#########################################################################')
        print('Start Training...', file=log_file)
        print('Start Training...')

        for total_epoch in range(TOTAL_EPOCH):
            kl_loss, r_loss = train_epoch(sess, encoder, generator, gen_data_loader, train_updates)
            if total_epoch % 5 == 0:
                generate_samples(sess, generator, batch_size, generated_num, z_size, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                print('epoch ', total_epoch, 'kl_loss: ', kl_loss, 'r_loss: ', r_loss, 'nll: ', test_loss,
                      file=log_file)
                print('epoch ', total_epoch, 'kl_loss: ', kl_loss, 'r_loss: ', r_loss, 'nll: ', test_loss)
                log_file.flush()

        log_file.close()


if __name__ == '__main__':
    main()
