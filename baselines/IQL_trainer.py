import os
import pickle

import gym
import d4rl2
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags

from jaxrl2.agents.pixel_iql import PixelIQLLearner
from jaxrl2.evaluation import evaluate

tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'randomized_kitchen_microwave-v1', 'Environment name.')
flags.DEFINE_string('save_dir', './IQL_full/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 250,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of training steps.')

flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    './baselines/configs/offline_pixels_config.py:iql',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

def main(_):
    from jax.lib import xla_bridge
    print('DEVICE:', xla_bridge.get_backend().platform)

    wandb.init(project='IQL_clean_v7')
    wandb.config.update(FLAGS)

    env = gym.make(FLAGS.env_name)
    env.seed(FLAGS.seed)

    from d4rl2.wrappers.kitchen_recorder import KitchenVideoRecorder
    eval_env = gym.make(FLAGS.env_name)
    eval_env.seed(FLAGS.seed + 42)
    eval_env = KitchenVideoRecorder(eval_env, os.path.join(FLAGS.save_dir, FLAGS.env_name, str(FLAGS.seed), 'eval_gifs'))

    #kwargs = dict(FLAGS.config)
    #if kwargs.pop('cosine_decay', False):
    #    kwargs['decay_steps'] = FLAGS.max_steps
    #agent = PixelIQLLearner(FLAGS.seed, env.observation_space.sample(),
    #                        env.action_space.sample(), **kwargs)

    print('Environment Cretaed')
    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = FLAGS.max_steps
    print(globals()[FLAGS.config.model_constructor])
    agent = globals()[FLAGS.config.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(),
        **kwargs)
    print('Agent created')

    replay_buffer = env.q_learning_dataset()
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(FLAGS.batch_size)

    print('Replay buffer loaded')

    print('Start training')
    for i in tqdm.tqdm(range(1, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = next(replay_buffer_iterator)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:

            if not FLAGS.tqdm:
                print(f"[IQL seed {FLAGS.seed}] {i}/{FLAGS.max_steps} steps")

            for k, v in update_info.items():
                if v.ndim == 0:
                    wandb.log({f'training/{k}': v}, step=i)
                    print(k, v)


        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes,
                                 progress_bar=True) ###===### ###---###
            for k, v in eval_info.items():
                wandb.log({f'evaluation/{k}': v}, step=i)

if __name__ == '__main__':
    app.run(main)
