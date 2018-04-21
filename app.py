import gym
import warnings
import argparse
import tensorflow as tf

from gym.envs import registry

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--environments', default='act', type=str,
	                                  help='Show a list of environments available')
parser.add_argument('--environment_name', default='MsPacman-v0', type=str,
	                                      help='The gym environment name')
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

def main(argv):
	args = parser.parse_args(argv[1:])

	if args.environments == 'list':
		for environment in registry.all():
			print(environment)
		# sys.close()
	elif args.environments == 'act':
		env = gym.make(args.environment_name)
		i_episodes = args.i_episodes
		timesteps = args.timesteps

		for i_episode in range(i_episodes):
			observation = env.reset()
			for t in range(timesteps):
				try:
					env.render()
					action = env.action_space.sample()
					stats['input']['actions'].append(action)
					observation, reward, done, info = env.step(action)
					stats['observations'].append(observation)
					stats['rewards'].append(reward)
					stats['output']['done'].append(done)
					stats['output']['info'].append(info)
					if done:
						max_episodes_range = (i_episodes - 1)
						episode_timesteps_iteration_limit = max_episodes_range - 1
						is_latest_episode = is_filled_latest_episode_with_iteration(i_episode, episode_timesteps_iteration_limit)
						increased_timestep = increase_timestep(t)
						print('i_episode {}'.format(i_episode))
						print('Episode finished after {} timesteps'.format(increased_timestep))
						stats['output']['timestep']['iteration'].append(t)
						stats['output']['timestep']['increased'].append(increased_timestep)
						print(check_output_env_label())
						break
				except Exception:
					print('Exception occured on the rendering execution')
	else:
		parser.print_help()

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)