import pickle
import random

import numpy as np

import configs
from target_lstm import TARGET_LSTM


def get_params_from_normal(shape):
    size = 1
    for item in shape:
        size *= item
    params = np.random.normal(loc=0.0, scale=1.0, size=size).astype(np.float32)
    return params.reshape(shape)


def main():
    config = configs.CONFIG_MAP['t_8']
    hparams = config.hparams

    SEED = config.seed
    START_TOKEN = config.start_token
    target_params_file = config.target_params_file

    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    batch_size = hparams.batch_size
    vocab_size = hparams.vocab_size
    feature_dim = hparams.emb_dim
    hidden_dim = hparams.hidden_dim
    max_seq_len = hparams.seq_len

    target_params = [
        get_params_from_normal([vocab_size, feature_dim]),
        get_params_from_normal([feature_dim, hidden_dim]),
        get_params_from_normal([hidden_dim, hidden_dim]),
        get_params_from_normal([hidden_dim]),
        get_params_from_normal([feature_dim, hidden_dim]),
        get_params_from_normal([hidden_dim, hidden_dim]),
        get_params_from_normal([hidden_dim]),
        get_params_from_normal([feature_dim, hidden_dim]),
        get_params_from_normal([hidden_dim, hidden_dim]),
        get_params_from_normal([hidden_dim]),
        get_params_from_normal([feature_dim, hidden_dim]),
        get_params_from_normal([hidden_dim, hidden_dim]),
        get_params_from_normal([hidden_dim]),
        get_params_from_normal([hidden_dim, vocab_size]),
        get_params_from_normal([vocab_size]), ]

    TARGET_LSTM(
        vocab_size, batch_size,
        feature_dim, hidden_dim,
        max_seq_len, START_TOKEN, target_params)  # The oracle model

    pickle.dump(target_params, open(target_params_file, 'wb'))


if __name__ == '__main__':
    main()
