
import gym
env = gym.make("SpaceInvaders-v0")
for i_episode in range(20):
	state = env.reset()
	for t in range(1000):
		env.render()
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			break
env.close()
