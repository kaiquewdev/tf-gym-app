import os
import gym
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from gym.envs import registry
from datetime import datetime
from collections import deque
from keras import backend as K
# from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
# from keras.optimizers import SGD
# from keras.optimizers import Adadelta
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

# from sklearn import model_selection

# has_ci_on_environ = 'CI' in os.environ
# is_ci_enabled = has_ci_on_environ and os.environ['CI'] == 'enabled'

# if not is_ci_enabled:
    # import gym_gomoku
    # import nesgym_super_mario_bros

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
parser.add_argument('--i_episodes', default=40, type=int, help='episodes')
# parser.add_argument('--timesteps', default=1000, type=int, help='playable timesteps')
parser.add_argument('--action_type', default='conditional', type=str,
                                     help='Kind of usage for action sample')
parser.add_argument('--seed_factor', default=2048, type=int, help='seed factor')
parser.add_argument('--render', default='present', type=str, help='rendering presence')
parser.add_argument('--episodes', default=10000, type=int, help='DQN Agent Episodes')
parser.add_argument('--pre_defined_state_size', default='gym', type=str,
                                                help='Observation shape based state size')
parser.add_argument('--i_seasons', default=2, type=int, help='Season iterated by episodes played')
parser.add_argument('--usage', type=str)

class DQNAgent:
    def __init__(self, state_size, action_size, timesteps):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.9902    # discount rate
        self.epsilon = 0.7  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0001
        self.timesteps = timesteps
        # self.learning_rate = 1e-4
        self.epochs = 2
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _mean_q(self, y_true, y_pred):
        return K.mean(K.max(y_pred, axis=-1))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(8, input_dim=self.state_size))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.timesteps))
        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss=self._huber_loss,
                      metrics=['accuracy'],
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.compile(optimizer='adam', loss='mse')
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # np.random.seed(1024)
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        es = EarlyStopping(monitor='val_loss', min_delta=2, patience=5, verbose=0, mode='auto')
        rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=5, min_lr=0.001)
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            reward = reward
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=self.epochs, verbose=0, callbacks=[es, rlrp])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class StatisticsInput(object):
    def __init__(self):
        self.actions = []

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
        self.input = StatisticsInput()
        self.output = StatisticsOutput()

statistics = Statistics()
stats = {'observations':statistics.observations,'rewards':statistics.rewards,
         'output':{'done':statistics.output.done,'info':statistics.output.info,
                   'timestep':{'iteration':statistics.output.timestep.iteration,
                               'increased':statistics.output.timestep.increased}},
         'input':{'actions':statistics.input.actions}}

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

    if args.usage == 'help':
        return parser.print_help()

    if is_environments_gen(args):
        _write_env_file(args)
    elif is_environments_list(args):
        all_registry = registry.all()
        registry_envs_name = [trim_env_spec_name(env.__repr__()) for env in all_registry]
        for environment in registry_envs_name:
            print(environment)
    elif is_environments_act(args):
        env = gym.make(args.environment_name)
        # if is_action_type('dqn', args):
        # if args.pre_defined_state_size == 'nesgym':
        #     pre_state_size = 172032
        # elif args.pre_defined_state_size == 'gym':
        #     pre_state_size = env.observation_space.shape[0]
        # elif args.pre_defined_state_size == 'gym-atari':
        #     pre_state_size = 100800
        # elif args.pre_defined_state_size == 'gym-atari-extend':
        #     pre_state_size = 120000
        # elif args.pre_defined_state_size == 'gym-atari-small':
        #     pre_state_size = 100800
        # elif args.pre_defined_state_size == 'gym-gomoku':
        #     pre_state_size = 361
        i_seasons = args.i_seasons
        def season():
            state_size = 100800
            action_size = env.action_space.n
            i_episodes = args.i_episodes
            timesteps = int((i_episodes**3)/2)
            agent = DQNAgent(state_size, action_size, timesteps)
            # try:
            #     agent.load('./weights/dqn_{}_{}_{}.h5'.format(args.environment_name.lower(), timesteps,
            #                                                   i_episodes))
            # except Exception:
            #     pass
            done = False
            batch_size = 32
            # timesteps = args.timesteps
            # factor = args.seed_factor
            for i_episode in range(i_episodes):
                print('Episodes: {}/{}'.format(i_episode + 1, i_episodes))
                # state_size = (1,) + env.observation_space.shape
                state = env.reset()
                # if is_action_type('dqn', args):
                state = np.reshape(state, [1, state_size])
                for t in range(timesteps):
                    env.render()
                    action = agent.act(state)
                    observation, reward, done, info = env.step(action)
                    reward = reward if not done else -10
                    observation = np.reshape(observation, [1, state_size])
                    agent.remember(state, action, reward, observation, done)
                    state = observation
                    if done:
                        break
                agent.replay(batch_size)
            agent.save('./weights/dqn_{}_{}_{}.h5'.format(args.environment_name.lower(), timesteps,
                                                          i_episodes))
        for i_season in range(i_seasons):
            print('Seasons: {}/{}'.format(i_season + 1, i_seasons))
            season()
        env.close()
    else:
        parser.print_help()

def run_main():
    tf.app.run(main)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    run_main()