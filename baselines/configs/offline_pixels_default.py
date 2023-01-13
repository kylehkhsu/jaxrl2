import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (32, 64, 128, 256)
    config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = 'VALID'
    config.latent_dim = 50

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.A_scaling = 3.0
    config.dropout_rate = config_dict.placeholder(float)
    config.cosine_decay = True

    config.tau = 0.005

    config.critic_reduction = 'mean'
    config.share_encoder = True

    return config
