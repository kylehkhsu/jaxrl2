import os
import pickle

import gym
import d4rl
import d4rl2
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags

from jaxrl2.agents.pixel_cql import PixelCQLLearner
from jaxrl2.evaluation import evaluate

from d4rl2.data import OfflineMemoryEfficientReplayBuffer
import copy
from gym.utils import seeding
import glob
import numpy as np
import collections

tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'randomized_kitchen_microwave-v1', 'Environment name.')
flags.DEFINE_string('save_dir', './CQL_full/', 'Tensorboard logging dir.')
flags.DEFINE_string('project', "CQL_clean_v7", 'WandB project.')
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
    './configs/offline_pixels_config.py:cql',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

def make_env(env_name):
    from jaxrl2.wrappers.combo import DeepMindControl, ActionRepeat, NormalizeActions, TimeLimit, RewardObs
    env = DeepMindControl(env_name, size=(64, 64))
    env = ActionRepeat(env, 2)
    env = NormalizeActions(env)
    env = TimeLimit(env, 1000 / 2)
    # env = RewardObs(env)
    return env

def WalkerDataGenerator(url: str, env: gym.Env, seed=None):

    observation_space = copy.deepcopy(env.observation_space.spaces)
    pixel_observation_space = observation_space.pop('pixels')
    num_stack = pixel_observation_space.shape[-1]

    np_random, seed = seeding.np_random(seed)
    while True:
        files = glob.glob(url + '/*.npz')
        np_random.shuffle(files)

        for file in files:
            # episode = np.load(file, allow_pickle=True)
            with open(file, 'rb') as f:
                episode = np.load(f, allow_pickle=True)
                episode = {k: episode[k] for k in episode.keys()}

            # episode['reward'] = np.sum([episode['reward_' + task] for
            #                            task in env.TASK_ELEMENTS], axis = 0)

            frames = collections.deque(maxlen=num_stack)
            for _ in range(num_stack):
                frames.append(episode["image"][0])
                    # np.concatenate([episode[cam + '_rgb'][0] for cam in env.cameras],
                    #                axis=-1))

            for t in range(episode['reward'].shape[0] - 1):
                transition = dict()
                transition['observations'] = dict()
                transition['observations']['pixels'] = np.stack(frames,
                                                                axis=-1)

                # transition['observations']['states'] = np.concatenate(
                #         [episode['robot_qp'][t],
                #          episode['ee_qp'][t],
                #          episode['ee_forces'][t]],
                #          axis=-1)

                transition['actions'] = episode['action'][t + 1]
                transition['rewards'] = episode['reward'][t + 1]

                frames.append(episode["image"][t + 1])
                    # np.concatenate([
                    #     episode[cam + '_rgb'][t + 1]  for cam in env.cameras],
                    #                axis=-1))

                transition['next_observations'] = dict()
                transition['next_observations']['pixels'] = np.stack(frames,
                                                                     axis=-1)

                # transition['next_observations']['states'] = np.concatenate(
                #         [episode['robot_qp'][t + 1],
                #          episode['ee_qp'][t + 1],
                #          episode['ee_forces'][t + 1]],
                #          axis=-1)

                transition['masks'] = 0.0
                transition['dones'] = 0.0

                yield transition

def walker_q_learning_dataset(env, datadir, include_pixels=False, size=50000):
    offline_generator = WalkerDataGenerator(
        datadir, env)
    dataset = OfflineMemoryEfficientReplayBuffer(
        env.observation_space,
        env.action_space,
        size,
        offline_generator,
        include_pixels=include_pixels)
    return dataset

def main(_):
    from jax.lib import xla_bridge
    print('DEVICE:', xla_bridge.get_backend().platform)

    # wandb.init(project='CQL_clean_v7')
    wandb.init(project=FLAGS.project)
    wandb.config.update(FLAGS)

    # env = gym.make(FLAGS.env_name)
    # env.seed(FLAGS.seed)
    #
    # from d4rl2.wrappers.kitchen_recorder import KitchenVideoRecorder
    # eval_env = gym.make(FLAGS.env_name)
    # eval_env.seed(FLAGS.seed + 42)
    # eval_env = KitchenVideoRecorder(eval_env, os.path.join(FLAGS.save_dir, FLAGS.env_name, str(FLAGS.seed), 'eval_gifs'))

    def wrap(env):
        from d4rl2.wrappers.frame_stack import FrameStack
        # assert FLAGS.num_stack == 1
        print(env.observation_space["pixels"].shape)
        env = FrameStack(env, num_stack=1)
        print(env.observation_space["pixels"].shape)
        env.reset()
        return env

    env = make_env(FLAGS.env_name)
    env = wrap(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    eval_env = make_env(FLAGS.env_name)
    eval_env = wrap(eval_env)
    eval_env.seed(FLAGS.seed + 42)

    # def wrap(env):
    #     from jaxrl2.wrappers import wrap_pixels
    #     if 'quadruped' in FLAGS.env_name:
    #         camera_id = 2
    #     else:
    #         camera_id = 0
    #     return wrap_pixels(env,
    #                        action_repeat=action_repeat,
    #                        image_size=FLAGS.image_size,
    #                        num_stack=FLAGS.num_stack,
    #                        camera_id=camera_id)

    # env = gym.make(FLAGS.env_name)
    # env = wrap(env)
    # env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    # env.seed(FLAGS.seed)
    #
    # eval_env = gym.make(FLAGS.env_name)
    # eval_env = wrap(eval_env)
    # eval_env.seed(FLAGS.seed + 42)

    print("env.reset()[\"pixels\"].shape:", env.reset()["pixels"].shape)

    #kwargs = dict(FLAGS.config)
    #if kwargs.pop('cosine_decay', False):
    #    kwargs['decay_steps'] = FLAGS.max_steps
    #agent = PixelIQLLearner(FLAGS.seed, env.observation_space.sample(),
    #                        env.action_space.sample(), **kwargs)

    print('Environment Created')
    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = FLAGS.max_steps
    print(globals()[FLAGS.config.model_constructor])
    agent = globals()[FLAGS.config.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(),
        **kwargs)
    print('Agent created')

    datadir = "/iris/u/khatch/preliminary_experiments/model_based_offline_online/LOMPO/data/walker/rafael_expert"
    replay_buffer = walker_q_learning_dataset(env, datadir)
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
                print(f"[CQL seed {FLAGS.seed}] {i}/{FLAGS.max_steps} steps")

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

"""
Make preemptible?
Change wanb logging project
Make it so that it's group wandb logging
Make tau 0?

Clip dataset actions when loading?

Make sure Victor's implementations are using the groups encoder

Double check proprio is called 'state', or rename stuff
"""
