import gym

env = gym.make('MsPacman-v0')

stats = {'observations':[],'rewards':[],
         'output':{'done':[],'info':[],'timestep':{'iteration':[],'increased':[]}},
         'input':{'actions':[]}}

def increase_timestep(t=int):
	return t + 1

def iterated_timesteps():
	key_check = 'increased'
	# has_output = 'output' in stats
	# has_timestep = has_output and ('timestep' in stats['output'])
	# has_timestep_key_check = has_timestep and (key_check in stats['output']['timestep'][key_check])
	# is_timestep_key_check_gt_zero = has_timestep_key_check and (stats['output']['timestep'][key_check] > 0)
	# pre_defined_output = (is_timestep_key_check_gt_zero and len(stats['output']['timestep'][key_check])) or 0
	pre_defined_output = len(stats['output']['timestep'][key_check])
	return pre_defined_output

def check_output_env_label():
	output_env_label = lambda: 'Episodes done with {}'.format(iterated_timesteps())
	return output_env_label()

def is_filled_latest_episode_with_iteration(i_episode_scoped, iteration_limit):
	return i_episode_scoped == iteration_limit

i_episodes = 10

for i_episode in range(i_episodes):
	observation = env.reset()
	for t in range(1000):
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