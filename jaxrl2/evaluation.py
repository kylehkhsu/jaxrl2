from typing import Dict

import gym
import numpy as np

from jaxrl2.data.dataset import Dataset

from tqdm import tqdm, trange


def evaluate(agent, env: gym.Env, num_episodes: int, progress_bar=False) -> Dict[str, float]: ###===### ###---###
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    ###===###
    # for _ in range(num_episodes):
    tbar = trange(num_episodes) if progress_bar else range(num_episodes)
    for _ in tbar:
    ###---###
        observation, done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)

    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}


def evaluate_log_prob(agent, dataset: Dataset, batch_size: int = 2048) -> float:
    num_iters = len(dataset) // batch_size
    total_log_prob = 0.0
    for j in range(num_iters):
        indx = np.arange(j * batch_size, (j + 1) * batch_size)
        batch = dataset.sample(batch_size, keys=("observations", "actions"), indx=indx)
        log_prob = agent.eval_log_probs(batch)
        total_log_prob += log_prob

    return total_log_prob / num_iters
