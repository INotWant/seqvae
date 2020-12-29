import sys

sys.path.append('../')

import pickle
import time

import numpy as np
import tensorflow as tf

import configs
from seq_vae_1.encoder import BidirectionalLstmEncoder
from seq_vae_1.generator import Generator
from seq_vae_1.rollout import ROLLOUT
from seqgan.dataloader import Gen_Data_loader, Dis_dataloader
from seqgan.discriminator import Discriminator
from target_lstm import TARGET_LSTM

# GPU's configuration
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True


# only use CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_z(batch_size, z_size):
    return np.random.multivariate_normal([0.] * z_size, np.eye(z_size), batch_size)


def generate_samples(sess, generator, batch_size, generated_num, z_size, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        z = get_z(batch_size, z_size)
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


def pre_train_epoch(sess, encoder, generator, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_kl_losses = []
    supervised_r_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        z, kl_loss = encoder.train_step(sess, batch)
        _, r_loss = generator.pretrain_step(sess, batch)
        supervised_kl_losses.append(kl_loss)
        supervised_r_losses.append(r_loss)
    return np.mean(supervised_kl_losses), np.mean(supervised_r_losses)


def main():
    time_str = str(int(time.time()))
    # configuration
    config = configs.CONFIG_MAP['t_8'] # t_64 t_8 t_20
    target_params_file = '../' + config.target_params_file
    start_token = config.start_token
    assert start_token == 0
    generated_num = config.generated_num
    hparams = config.hparams

    TOTAL_BATCH = 300
    batch_size = hparams.batch_size
    seq_len = hparams.seq_len
    z_size = hparams.z_size
    dropout_keep_prob = hparams.dropout_keep_prob
    rollout_num = hparams.rollout_num
    positive_file = '../' + config.positive_file_txt
    negative_file = '../save/seq_vae_1_' + time_str + '_' + config.negative_file_txt
    eval_file = '../save/eval_file.txt'

    gen_data_loader = Gen_Data_loader(batch_size, seq_len)
    likelihood_data_loader = Gen_Data_loader(batch_size, seq_len)  # For testing
    dis_data_loader = Dis_dataloader(batch_size, seq_len)

    # log file
    hparams_str = hparams.__str__()
    log_fn = '../save/seq_vae_1_' + time_str + '.log'
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
        discriminator = Discriminator(
            sequence_length=hparams.seq_len,
            num_classes=2,
            vocab_size=hparams.vocab_size,
            embedding_size=hparams.emb_dim,
            filter_sizes=hparams.dis_filter_sizes,
            num_filters=hparams.dis_num_filters,
            l2_reg_lambda=0.2,
            learning_rate=hparams.dis_learning_rate)

        gen_data_loader.create_batches(positive_file)
        sess.run(tf.global_variables_initializer())

        #  pre-train generator
        print('Start pre-training...', file=log_file)
        print('Start pre-training...')
        for epoch in range(40):
            kl_loss, r_loss = pre_train_epoch(sess, encoder, generator, gen_data_loader)
            if epoch % 5 == 0:
                generate_samples(sess, generator, batch_size, generated_num, z_size, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                print('pre-train epoch ', epoch, 'kl_loss: ', kl_loss, 'r_loss: ', r_loss, 'nll: ', test_loss,
                      file=log_file)
                print('pre-train epoch ', epoch, 'kl_loss: ', kl_loss, 'r_loss: ', r_loss, 'nll: ', test_loss)
                log_file.flush()

        print('Start pre-training discriminator...', file=log_file)
        print('Start pre-training discriminator...')
        # Train 3 epoch on the generated data and do this for 50 times
        for _ in range(40):
            generate_samples(sess, generator, batch_size, generated_num, z_size, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)

        rollout = ROLLOUT(generator, hparams.dec_update_rate)

        print('#########################################################################', file=log_file)
        print('#########################################################################')
        print('Start Training...', file=log_file)
        print('Start Training...')

        for total_batch in range(TOTAL_BATCH):
            # Train the generator for one step
            kl_loss = ''
            for it in range(1):
                batch = gen_data_loader.next_batch()
                z, kl_loss = encoder.train_step(sess, batch)
                samples = generator.generate(sess, z)
                rewards = rollout.get_reward(sess, samples, z, rollout_num, discriminator)
                _, pg_loss = generator.train_step(sess, samples, rewards)

            # Test
            if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
                generate_samples(sess, generator, batch_size, generated_num, z_size, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                print('epoch: ', total_batch, 'kl_loss: ', kl_loss, 'pg_loss: ', pg_loss, 'nll: ', test_loss,
                      file=log_file)
                print('epoch: ', total_batch, 'kl_loss: ', kl_loss, 'pg_loss: ', pg_loss, 'nll: ', test_loss)

            # Update roll-out parameters
            rollout.update_params()

            # Train the discriminator
            for _ in range(hparams.dis_train_freq):
                generate_samples(sess, generator, batch_size, generated_num, z_size, negative_file)
                dis_data_loader.load_train_data(positive_file, negative_file)

                for _ in range(3):
                    dis_data_loader.reset_pointer()
                    for _ in range(dis_data_loader.num_batch):
                        x_batch, y_batch = dis_data_loader.next_batch()
                        feed = {
                            discriminator.input_x: x_batch,
                            discriminator.input_y: y_batch,
                            discriminator.dropout_keep_prob: dropout_keep_prob
                        }
                        _ = sess.run(discriminator.train_op, feed)

            log_file.flush()
        log_file.close()


if __name__ == '__main__':
    main()
