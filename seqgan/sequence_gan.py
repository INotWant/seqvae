import pickle
import sys
import time

sys.path.append('../')

import numpy as np
import tensorflow as tf

import configs
from seqgan.dataloader import Gen_Data_loader, Dis_dataloader
from seqgan.discriminator import Discriminator
from seqgan.generator import Generator
from seqgan.rollout import ROLLOUT
from target_lstm import TARGET_LSTM

# GPU's configuration
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True


# only use CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

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


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    time_str = str(int(time.time()))
    # configuration
    config = configs.CONFIG_MAP['t_64'] # t_8 t_20
    target_params_file = '../' + config.target_params_file
    start_token = config.start_token
    assert start_token == 0
    generated_num = config.generated_num
    hparams = config.hparams

    TOTAL_BATCH = 150
    batch_size = hparams.batch_size
    seq_len = hparams.seq_len
    dropout_keep_prob = hparams.dropout_keep_prob
    rollout_num = hparams.rollout_num
    vocab_size = hparams.vocab_size

    positive_file = '../' + config.positive_file_txt
    negative_file = '../save/seq_gan_' + time_str + '_' + config.negative_file_txt
    eval_file = '../save/eval_file.txt'

    gen_data_loader = Gen_Data_loader(batch_size, seq_len)
    likelihood_data_loader = Gen_Data_loader(batch_size, seq_len)  # For testing
    dis_data_loader = Dis_dataloader(batch_size, seq_len)

    # log file
    hparams_str = hparams.__str__()
    log_fn = '../save/seq_gan_' + time_str + '.log'
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

        generator = Generator(
            vocab_size,
            batch_size,
            hparams.emb_dim,
            hparams.hidden_dim,
            hparams.seq_len,
            start_token,
            learning_rate=hparams.dencoder_learning_rate)

        discriminator = Discriminator(
            sequence_length=hparams.seq_len,
            num_classes=2, vocab_size=vocab_size,
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
        for epoch in range(120):
            pre_train_epoch(sess, generator, gen_data_loader)
            if epoch % 5 == 0:
                generate_samples(sess, generator, batch_size, generated_num, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                print('pre-train epoch ', epoch, 'test_loss ', test_loss)
                buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
                log_file.write(buffer)
                log_file.flush()

        print('Start pre-training discriminator...', file=log_file)
        print('Start pre-training discriminator...')
        # Train 3 epoch on the generated data and do this for 50 times
        for _ in range(50):
            generate_samples(sess, generator, batch_size, generated_num, negative_file)
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
        print('Start Adversarial Training...', file=log_file)
        print('Start Adversarial Training...')
        for total_batch in range(TOTAL_BATCH):
            # Train the generator for one step
            for it in range(1):
                samples = generator.generate(sess)
                rewards = rollout.get_reward(sess, samples, rollout_num, discriminator)
                feed = {generator.x: samples, generator.rewards: rewards}
                _ = sess.run(generator.g_updates, feed_dict=feed)

            # Test
            if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
                generate_samples(sess, generator, batch_size, 100, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                print('epoch: ', total_batch, 'nll: ', test_loss, file=log_file)
                print('epoch: ', total_batch, 'nll: ', test_loss)

            # Update roll-out parameters
            rollout.update_params()

            # Train the discriminator
            for _ in range(hparams.dis_train_freq):
                generate_samples(sess, generator, batch_size, generated_num, negative_file)
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
