import gym
import warnings
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from gym.envs import registry

from datetime import datetime

from dqn import deque
from dqn import Sequential
from dqn import DQNAgent

import os

import gym_pull as pulling

has_ci_on_environ = 'CI' in os.environ
is_ci_enabled = has_ci_on_environ and os.environ['CI'] == 'enabled'

if not is_ci_enabled:
	print('CI disabled')
	import gym_gomoku
	import nesgym_super_mario_bros

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--environments', default='act', type=str,
	                                  help='Show a list of environments available')
parser.add_argument('--env_name', default='pacman', type=str,
								  help='Generated environment name')
parser.add_argument('--environment_name', default='MsPacman-v0', type=str,
	                                      help='The gym environment name')
parser.add_argument('--output_stats_filename', type=str,
	                                           help='Statistics about turn saved on a csv file')
parser.add_argument('--i_episodes', default=10, type=int, help='episodes')
parser.add_argument('--timesteps', default=1000, type=int, help='playable timesteps')
parser.add_argument('--action_type', default='conditional', type=str,
	                                 help='Kind of usage for action sample')
parser.add_argument('--seed_factor', default=2048, type=int, help='seed factor')
parser.add_argument('--render', default='present', type=str, help='rendering presence')
parser.add_argument('--episodes', default=10000, type=int, help='DQN Agent Episodes')
parser.add_argument('--pre_defined_state_size', default='gym', type=str,
	                                            help='Observation shape based state size')
parser.add_argument('--pull', type=str,
							  help='Name of game to pull')

stats = {'observations':[],'rewards':[],
         'output':{'done':[],'info':[],
         		   'timestep':{'iteration':[],'increased':[]}},
         'input':{'actions':[]}}

class StatisticsInput(object):
	pass

class StatisticsOutputTimestep(object):
	def __init__(self):
		self.iteration = []
		self.increased = []

class StatisticsOutput(object):
	def __init__(self):
		self.done = []
		self.info = []
		self.timestep = StatisticsOutputTimestep()

class Statistics(object):
	def __init__(self):
		self.observations = []
		self.rewards = []
		self.output = StatisticsOutput()

def increase_timestep(t=int):
	return t + 1

def iterated_timesteps(key_check='increased'):
	# has_output = 'output' in stats
	# has_timestep = (has_output and ('timestep' in stats['output'])) or 0
	# has_timestep_key_check = (has_timestep and (key_check in stats['output']['timestep'][key_check])) or 0
	# is_timestep_key_check_gt_zero = (has_timestep_key_check and (stats['output']['timestep'][key_check] > 0)) or 0
	# pre_defined_output = (is_timestep_key_check_gt_zero and len(stats['output']['timestep'][key_check])) or 0
	pre_defined_output = len(stats['output']['timestep'][key_check])
	return pre_defined_output

def check_output_env_label():
	return 'Episodes done with {}'.format(iterated_timesteps())

def is_filled_latest_episode_with_iteration(i_episode_scoped, iteration_limit):
	return i_episode_scoped == iteration_limit

def collect_stat(v,props,stash):
	if len(props) == 1:
		curr = stash[props[0]]
		curr.append(v)
		return curr
	elif len(props) == 2:
		curr = stash[props[0]][props[1]]
		curr.append(v)
		return curr
	elif len(props) == 3:
		curr = stash[props[0]][props[1]][props[2]]
		curr.append(v)
		return curr
	return []

def composed_sample(s=2, vm=None):
	if vm:
		gen_sample = lambda: vm.action_space.sample()
		gen_list_based_sample = lambda subdued_limit: [gen_sample() for _ in
		                                               range(subdued_limit)] 
		return gen_list_based_sample(s)
	return []

def random_action_space_sample_choice(s=2, vm=None, factor=1024):
	np.random.seed(factor)
	if vm:
		choices = composed_sample(s,vm)
		limited_index = len(choices) - 1
		choice_index = np.random.randint(limited_index)
		return choices[choice_index]
	return -1

def trim_env_spec_name(k):
	return k.split('(')[1][:-1]

def is_action_type(name, args_scoped):
	return args_scoped.action_type == name

def is_environments_name(name, args_scoped):
	return args_scoped.environments == name

def is_environments_list(args_scoped):
	return is_environments_name('list', args_scoped)

def is_environments_act(args_scoped):
	return is_environments_name('act', args_scoped)

def is_environments_gen(args_scoped):
	return is_environments_name('gen', args_scoped)

def is_environments_pull(args_scoped):
	return is_environments_name('pull', args_scoped)

file_content = '''
import gym
env = gym.make("%s")
for i_episode in range(10):
	state = env.reset()
	for t in range(1000):
		env.render()
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			break
env.close()
'''

def _write_env_file(args_scoped):
	env_name = args_scoped.env_name
	environment_name = args_scoped.environment_name
	label = 'envs/{}gym-env.py'
	label_format = '-'.join([(env_name),''])
	with open(label.format(label_format), 'w') as f:
		f.write(file_content % ((environment_name)))
		f.close()
		print('Gym environment file created!')

def main(argv):
	args = parser.parse_args(argv[1:])

	if is_environments_gen(args):
		_write_env_file(args)
	elif is_environments_list(args):
		all_registry = registry.all()
		registry_envs_name = [trim_env_spec_name(env.__repr__()) for env in all_registry]
		for environment in registry_envs_name:
			print(environment)
	elif is_environments_act(args):
		env = gym.make(args.environment_name)
		if is_action_type('dqn', args):
			if args.pre_defined_state_size == 'nesgym':
				state_size = 172032
			elif args.pre_defined_state_size == 'gym':
				state_size = env.observation_space.shape[0]
			elif args.pre_defined_state_size == 'gym-atari':
				state_size = 100800
			elif args.pre_defined_state_size == 'gym-atari-extend':
				state_size = 120000
			elif args.pre_defined_state_size == 'gym-gomoku':
				state_size = 361
			action_size = env.action_space.n
			agent = DQNAgent(state_size, action_size)
			done = False
			batch_size = 32
		i_episodes = args.i_episodes
		timesteps = args.timesteps
		factor = args.seed_factor
		for i_episode in range(i_episodes):
			state = env.reset()
			if is_action_type('dqn', args):
				state = np.reshape(state, [1, state_size])
			for t in range(timesteps):
				try:
					if args.render == 'present': env.render()
					if args.render == 'presented': env.render(args.render)
					if args.action_type == 'alternate':
						action_choice = i_episodes*2
						action = random_action_space_sample_choice(action_choice, env, factor)
					elif args.action_type == 'specific':
						action = env.action_space.sample()
					elif args.action_type == 'conditional':
						action_choice = i_episodes
						action = random_action_space_sample_choice(action_choice, env, factor)
					elif args.action_type == 'numerical':
						action = env.action_space.n
					elif is_action_type('dqn', args):
						action = agent.act(state)
					collect_stat(action,['input','actions'],stats)
					observation, reward, done, info = env.step(action)
					if is_action_type('dqn', args):
						reward = reward if not done else -10
						observation = np.reshape(observation, [1, state_size])
						agent.remember(state, action, reward, observation, done)
						state = observation
					# collect_stat(observation,['observation'],stats)
					collect_stat(reward,['rewards'],stats)
					# collect_stat(done,['output','done'],stats)
					# collect_stat(info,['output','info'],stats)
					if done:
						max_episodes_range = (i_episodes - 1)
						episode_timesteps_iteration_limit = max_episodes_range - 1
						is_latest_episode = is_filled_latest_episode_with_iteration(i_episode, episode_timesteps_iteration_limit)
						increased_timestep = increase_timestep(t)
						print('i_episode {}'.format(i_episode))
						print('Episode finished after {} timesteps'.format(increased_timestep))
						if is_action_type('dqn', args):
							print('Episode: {}/{}, score: {}, e: {:.2}'
								  .format(i_episode, i_episodes, t, agent.epsilon))
						collect_stat(t,['output','timestep','iteration'],stats)
						collect_stat(increased_timestep,['output','timestep','increased'],stats)
						is_latest_episode_to_save_state = lambda args_cached: is_latest_episode and args_cached.output_stats_filename
						if is_latest_episode_to_save_state(args):
							filename = args.output_stats_filename
							pre_df = {
								# 'observations': stats['observations'],
								'rewards': stats['rewards'],
								# 'done-output': stats['output']['done'],
								# 'info-output': stats['output']['info'],
								# 'iteration-timestep': stats['output']['timestep']['iteration'],
								# 'increased-timestep': stats['output']['timestep']['increased'],
								'actions-input': stats['input']['actions']
							}
							df = pd.DataFrame(pre_df)
							stamp = lambda: '%s' % (int(datetime.now().timestamp()))
							with open('data/{}-{}.csv'.format(stamp(),filename),'w') as f:
								f.write(df.to_csv())
								f.close()
							print('Statistics file saved ({}.csv)!'.format(filename))
							del df
							del filename
						print(check_output_env_label())
						del is_latest_episode_to_save_state
						del increased_timestep
						del is_latest_episode
						del episode_timesteps_iteration_limit
						del max_episodes_range
						break
				except Exception as e:
					print('Rendering execution ({})'.format(e))
				finally:
					print('Execution of timestep done')
			if is_action_type('dqn', args) and (len(agent.memory) > batch_size):
				agent.replay(batch_size)
		env.close()
	else:
		parser.print_help()

def run_main():
	tf.app.run(main)

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	run_main()