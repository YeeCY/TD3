import gym.wrappers
import random
import numpy as np
import torch

import metaworld
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


class SingleMT1Wrapper(gym.Wrapper):
	def __init__(self, env, tasks):
		assert isinstance(env, SawyerXYZEnv), f"Invalid environment type: {type(env)}"
		super().__init__(env)

		self.tasks = tasks

	def _set_random_task(self):
		task = random.choice(self.tasks)
		self.env.set_task(task)

	def reset(self, random_task=True):
		if random_task:
			self._set_random_task()

		return self.env.reset()

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		if self.env.curr_path_length > self.env.max_path_length or info.get('success'):
			done = True

		return obs, reward, done, info


class TaskNameWrapper(gym.Wrapper):
	"""Add task_name or task_id to environment infos.
    Args:
        env (gym.Env): The environment to wrap.
        task_name (str or None): Task name to be added, if any.
        task_id (int or None): Task ID to be added, if any.
    """

	def __init__(self, env, task_name=None, task_id=None):
		super().__init__(env)
		self._task_name = task_name
		self._task_id = task_id

	def step(self, action):
		"""gym.Env step for the active task environment.
        Args:
            action (np.ndarray): Action performed by the agent in the
                environment.
        Returns:
            tuple:
                np.ndarray: Agent's observation of the current environment.
                float: Amount of reward yielded by previous action.
                bool: True iff the episode has ended.
                dict[str, np.ndarray]: Contains auxiliary diagnostic
                    information about this time-step.
        """
		# (chongyi zheng): remove 'es'
		obs, reward, done, info = super().step(action)
		if self._task_name is not None:
			info['task_name'] = self._task_name
		if self._task_id is not None:
			info['task_id'] = self._task_id
		return obs, reward, done, info


class TimeLimit(gym.Wrapper):
	def __init__(self, env, max_episode_steps=None):
		super(TimeLimit, self).__init__(env)
		if max_episode_steps is None and self.env.spec is not None:
			max_episode_steps = env.spec.max_episode_steps
		if self.env.spec is not None:
			self.env.spec.max_episode_steps = max_episode_steps
		self._max_episode_steps = max_episode_steps
		self._elapsed_steps = None

	def step(self, action):
		assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
		observation, reward, done, info = self.env.step(action)
		self._elapsed_steps += 1
		if self._elapsed_steps >= self._max_episode_steps:
			info['TimeLimit.truncated'] = not done
			done = True
		else:
			# (chongyi zheng): add this to make number of attributes in info compatible at each timestep
			info['TimeLimit.truncated'] = done
			done = False

		return observation, reward, done, info

	def reset(self, **kwargs):
		self._elapsed_steps = 0
		return self.env.reset(**kwargs)


def set_seed_everywhere(seed):
	# Seed python RNG
	random.seed(seed)
	# Seed numpy RNG
	np.random.seed(seed)
	# Seed torch
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def make_single_metaworld_env(env_name, seed=None):
	mt1 = metaworld.MT1(env_name, seed=seed)
	env = mt1.train_classes[env_name]()
	# task = random.choice(mt1.train_tasks)
	# environment.set_task(task)
	# env = SingleMT1Wrapper(env, mt1.train_tasks)
	env.set_task(mt1.train_tasks[0])

	# env = TaskNameWrapper(env, task_name=env_name)
	env = TimeLimit(env, max_episode_steps=env.max_path_length)

	env.seed(seed)

	return env
