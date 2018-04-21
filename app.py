import gym
import warnings
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from gym.envs import registry

from datetime import datetime

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--environments', default='act', type=str,
	                                  help='Show a list of environments available')
parser.add_argument('--environment_name', default='MsPacman-v0', type=str,
	                                      help='The gym environment name')
parser.add_argument('--output_stats_filename', type=str,
	                                           help='Statistics about turn saved on a csv file')
parser.add_argument('--i_episodes', default=10, type=int, help='episodes')
parser.add_argument('--timesteps', default=1000, type=int, help='playable timesteps')

stats = {'observations':[],'rewards':[],
         'output':{'done':[],'info':[],
         		   'timestep':{'iteration':[],'increased':[]}},
         'input':{'actions':[]}}

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

def random_action_space_sample_choice(s=2, vm=None):
	if vm:
		choices = composed_sample(s,vm)
		limited_index = len(choices) - 1
		choice_index = np.random.randint(limited_index)
		return choices[choice_index]
	return -1

def main(argv):
	args = parser.parse_args(argv[1:])

	is_environments_name = lambda name, args_scoped: args_scoped.environments == name
	is_environments_list = lambda args_scoped: is_environments_name('list', args_scoped)
	is_environments_act = lambda args_scoped: is_environments_name('act', args_scoped)

	if is_environments_list(args):
		for environment in registry.all():
			print(environment)
	elif is_environments_act(args):
		env = gym.make(args.environment_name)
		i_episodes = args.i_episodes
		timesteps = args.timesteps
		for i_episode in range(i_episodes):
			observation = env.reset()
			for t in range(timesteps):
				try:
					env.render()
					action = random_action_space_sample_choice(10, env)
					collect_stat(action,['input','actions'],stats)
					observation, reward, done, info = env.step(action)
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
						break
				except Exception as e:
					print('Rendering execution ({})'.format(e))
				finally:
					print('Execution of timestep done')
	else:
		parser.print_help()

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)