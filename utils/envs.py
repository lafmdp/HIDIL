'''
  Generate different type of dynamics mismatch.
  @python version : 3.6.4
'''

import gym

from utils.utils import *


def get_new_gravity_env(variety, env_name):
    update_target_env_gravity(variety, env_name)
    env = gym.make(env_name)

    return env


def get_source_env(env_name="Walker2d-v2"):
    update_source_env(env_name)
    env = gym.make(env_name)

    return env


def get_new_density_env(variety, env_name):
    update_target_env_density(variety, env_name)
    env = gym.make(env_name)

    return env


def get_new_friction_env(variety, env_name):
    update_target_env_friction(variety, env_name)
    env = gym.make(env_name)

    return env
