import pickle
import random

import numpy as np
import tensorflow as tf

import configs
from target_lstm import TARGET_LSTM


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    np.save(output_file, generated_samples)


def main():
    config = configs.CONFIG_MAP['t_64']
    hparams = config.hparams

    SEED = config.seed
    START_TOKEN = config.start_token
    target_params_file = config.target_params_file
    positive_file = config.positive_file
    generated_num = config.generated_num

    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    batch_size = hparams.batch_size
    vocab_size = hparams.vocab_size
    emb_dim = hparams.emb_dim
    hidden_dim = hparams.hidden_dim
    seq_len = hparams.seq_len

    target_params = pickle.load(open(target_params_file, 'rb'))

    target_lstm = TARGET_LSTM(
        vocab_size, batch_size,
        emb_dim, hidden_dim,
        seq_len, START_TOKEN, target_params)  # The oracle model

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    generate_samples(sess, target_lstm, batch_size, generated_num, positive_file)


def txt_to_np(txt_fn, np_fn, seq_len):
    token_stream = []
    with open(txt_fn, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            if len(parse_line) == seq_len:
                token_stream.append(parse_line)
    np.save(np_fn, token_stream)


def np_to_txt(np_fn, txt_fn):
    data = np.load(np_fn)
    with open(txt_fn, 'w') as f:
        for row in data:
            f.write(' '.join(str(ele) for ele in row))
            f.write('\n')


if __name__ == '__main__':
    main()

    config = configs.CONFIG_MAP['t_64']
    positive_file = config.positive_file
    positive_file_txt = config.positive_file_txt
    np_to_txt(positive_file, positive_file_txt)
