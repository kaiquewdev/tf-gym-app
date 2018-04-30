import tensorflow as tf
import numpy as np
import gym
import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import warnings; warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--np_random_seed', default=128, type=int,
                                        help='Random seed factor')
parser.add_argument('--environment_seed', default=128, type=int,
                                          help='Environment seed factor')
parser.add_argument('--environment_name', default='MsPacman-v0', type=str,
                                          help='The gym environment name')
parser.add_argument('--memory_limit', default=50000, type=int,
                                      help='Sequential memory limit')
parser.add_argument('--warmup', default=10, type=int,
                                help='Warmup the steps')
parser.add_argument('--nb_steps', default=50000, type=int,
                                  help='Steps used on the game completition')
# parser.add_argument('--batch_size', default=64, type=int,
#                                     help='Nested iteration process across the model fitting')
parser.add_argument('--nb_episodes', default=10, type=int,
                                     help='Goal interchangeble process limit')
# parser.add_argument('--log_dir', default='./dqn_logs', type=str,
#                                  help='Logging directory output')

def main(argv):
    args = parser.parse_args(argv[1:])

    ENV_NAME = args.environment_name
    NP_RANDOM_SEED = args.np_random_seed
    ENV_SEED = args.environment_seed
    MEMORY_LIMIT = args.memory_limit
    WARMUP = args.warmup
    NB_STEPS = args.nb_steps
    # BATCH_SIZE = args.batch_size
    NB_EPISODES = args.nb_episodes
    # LOG_DIR = args.log_dir

    TARGET_LEARNING_RATE = 1e-2
    OPTIMIZER_LEARNING_RATE = 1e-3

    # tensorboard = TensorBoard(log_dir=LOG_DIR, histogram_freq=0,
    #                           batch_size=BATCH_SIZE, write_graph=True,
    #                           write_grads=False, write_images=False,
    #                           embeddings_freq=0, embeddings_layer_names=None,
    #                           embeddings_metadata=None)

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(NP_RANDOM_SEED)
    env.seed(ENV_SEED)
    nb_actions = env.action_space.n
    w_format = './weights/{}_{}_{}_{}'

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    model.load_weights(w_format.format('dqn', ENV_NAME,
                                       NB_EPISODES, NB_STEPS))

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions,
                   memory=memory, nb_steps_warmup=WARMUP,
                   target_model_update=TARGET_LEARNING_RATE, policy=policy)

    dqn.compile(Adam(lr=OPTIMIZER_LEARNING_RATE), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=NB_STEPS, 
            visualize=True, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights(w_format.format('dqn', ENV_NAME,
                                     NB_EPISODES, NB_STEPS), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=NB_EPISODES, visualize=True)

def run_main():
    tf.app.run(main)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    run_main()