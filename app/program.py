import argparse
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
parser.add_argument('--usage', type=str)