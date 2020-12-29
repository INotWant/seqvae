import collections

from tensorflow.contrib.training import HParams


class Config(collections.namedtuple(
    'Config',
    ['seed',
     'start_token',
     'generated_num',
     'hparams',
     'target_params_file',
     'positive_file',
     'positive_file_txt',
     'negative_file_txt',
     ], )):

    def values(self):
        return self._asdict()


Config.__new__.__defaults__ = (None,) * len(Config._fields)

CONFIG_MAP = {
    't_8': Config(
        seed=88,
        start_token=0,
        generated_num=10000,

        target_params_file='save/target_params.pkl',
        positive_file='save/real_data_8.npy',
        positive_file_txt='save/real_data_8.txt',
        negative_file_txt='generator_sample_8.txt',

        hparams=HParams(
            batch_size=32,
            z_size=64,
            seq_len=8,
            emb_dim=64,
            vocab_size=5000,
            hidden_dim=32,

            # seq_vae
            dropout_keep_prob=0.75,

            # encoder
            free_bits=0,
            encoder_learning_rate=0.01,
            enc_rnn_size=[32, 32],
            residual_encoder=True,

            # generator or decoder
            dencoder_learning_rate=0.01,
            compute_rewards_step=1,  # for seq_vae_2
            dec_update_rate=0.80,
            rollout_num=16,

            # discriminator
            dis_learning_rate=1e-4,
            dis_train_freq=5,
            dis_filter_sizes=[1, 2, 3, 4, 6, 8],
            dis_num_filters=[100, 200, 100, 200, 200, 100],
        ), ),

    't_20': Config(
        seed=88,
        start_token=0,
        generated_num=10000,

        target_params_file='save/target_params.pkl',
        positive_file='save/real_data_20.npy',
        positive_file_txt='save/real_data_20.txt',
        negative_file_txt='generator_sample_20.txt',

        hparams=HParams(
            batch_size=32,
            z_size=64,
            seq_len=20,
            emb_dim=64,
            vocab_size=5000,
            hidden_dim=32,

            # seq_vae
            dropout_keep_prob=0.75,

            # encoder
            free_bits=0,
            encoder_learning_rate=0.01,
            enc_rnn_size=[32, 32],
            residual_encoder=True,

            # generator or decoder
            dencoder_learning_rate=0.01,
            compute_rewards_step=1,  # for seq_vae_2
            dec_update_rate=0.80,
            rollout_num=16,

            # discriminator
            dis_learning_rate=1e-4,
            dis_train_freq=5,
            dis_filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
            dis_num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
        ), ),

    't_64': Config(
        seed=88,
        start_token=0,
        generated_num=10000,

        target_params_file='save/target_params.pkl',
        positive_file='save/real_data_64.npy',
        positive_file_txt='save/real_data_64.txt',
        negative_file_txt='generator_sample_64.txt',

        hparams=HParams(
            batch_size=32,
            z_size=64,
            seq_len=64,
            emb_dim=64,
            vocab_size=5000,
            hidden_dim=32,

            # seq_vae
            dropout_keep_prob=0.75,

            # encoder
            free_bits=0,
            encoder_learning_rate=0.01,
            enc_rnn_size=[32, 32],
            residual_encoder=True,

            # generator or decoder
            dencoder_learning_rate=0.01,
            compute_rewards_step=1,  # for seq_vae_2
            dec_update_rate=0.80,
            rollout_num=8,

            # discriminator
            dis_learning_rate=1e-4,
            dis_train_freq=5,
            dis_filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 48, 64],
            dis_num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 160, 160],
        ), ),
}
