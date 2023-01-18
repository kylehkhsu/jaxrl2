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

from jaxrl2.agents.pixel_cql import PixelCQLLearner
from jaxrl2.evaluation import evaluate

tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'randomized_kitchen_microwave-v1', 'Environment name.')
flags.DEFINE_string('save_dir', './CQL_full/', 'Tensorboard logging dir.')
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


# import mujoco_py

class MetaWorldEnv:

  def __init__(self, name="assembly-v2", action_repeat=4, size=(64, 64)):
      # import sys; sys.path.append("/iris/u/khatch/anaconda3/envs/combo/lib/python3.7/site-packages/")
      from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
      render_params={"assembly-v2" : {"elevation": -22.5,
                                      "azimuth": 15,
                                      "distance": 0.5,
                                      "lookat": np.array([-0.15, 0.65, 0.25])}}

      self._env = ALL_V2_ENVIRONMENTS[name]()
      self._env.max_path_length = np.inf
      self._env._freeze_rand_vec = False
      self._env._partially_observable = False
      self._env._set_task_called = True

      self.hand_init_pose = self._env.hand_init_pos.copy()

      self.action_repeat = action_repeat

      self.size = size
      self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)

      self.set_viewer_params(render_params[name])

  def __getattr__(self, attr):
     if attr == '_wrapped_env':
       raise AttributeError()
     return getattr(self._env, attr)

  @property
  def observation_space(self):
        spaces = {}
        spaces['image'] = gym.spaces.Box(0, 255, (self.size[0], self.size[1], 3), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

  def set_viewer_params(self, params):
      self.viewer.cam.elevation = params["elevation"]
      self.viewer.cam.azimuth = params["azimuth"]
      self.viewer.cam.distance = params["distance"]
      self.viewer.cam.lookat[:] = params["lookat"][:]

  def step(self, action):
    print("Env action", action)
    action = action.copy()
    action[:3] += 0.050 * np.random.normal(size=(3,))
    action[-1] += 0.025 * np.random.normal()
    action = np.clip(action, -0.99, 0.99)

    reward = 0.0
    for _ in range(self.action_repeat):
        state, rew, done, info = self._env.step(action)
        reward += rew
        if done:
            break
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    obs = {'image':img, 'state':state, 'reward dense':reward, 'reward sparse':1.0 * info['success']}
    return obs, reward, done, info

  def reset(self):
    self._env.hand_init_pos = self.hand_init_pose + 0.05 * np.random.normal(size = 3)
    _ = self._env.reset()
    state,_,_,_ = self._env.step(np.zeros(self.action_space.shape))
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    obs = {'image':img, 'state':state, 'reward dense':0.0, 'reward sparse':0.0}
    return obs

  def render(self, mode, width = 128, height = 128):
      self.viewer.render(width=width, height=width)
      img = self.viewer.read_pixels(width, height, depth=False)
      img = img[::-1]
      return img

def main(_):
    from jax.lib import xla_bridge
    print('DEVICE:', xla_bridge.get_backend().platform)

    wandb.init(project='CQL_clean_v7')
    wandb.config.update(FLAGS)

    # env = gym.make(FLAGS.env_name)
    # env.seed(FLAGS.seed)
    #
    # from d4rl2.wrappers.kitchen_recorder import KitchenVideoRecorder
    # eval_env = gym.make(FLAGS.env_name)
    # eval_env.seed(FLAGS.seed + 42)
    # eval_env = KitchenVideoRecorder(eval_env, os.path.join(FLAGS.save_dir, FLAGS.env_name, str(FLAGS.seed), 'eval_gifs'))

    env = MetaWorldEnv()
    import pdb; pdb.set_trace()

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
