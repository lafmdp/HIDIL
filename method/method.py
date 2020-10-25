'''
  Base class for different methods.
  @python version : 3.6.4
'''

import numpy as np


class method_base():

    def __init__(self):
        pass

    def get_action(self, s):
        pass

    def set_state(self, s):
        pass

    def get_std(self):
        return 0


class ours(method_base):

    def __init__(self, simulator, sim_policy, gcp, D, horizon):
        """

        :param simulator: a simulator we train our policy
        :param gcp: goal conditioned policy
        :param sim_policy: policy trained in simulator
        :param D: D(s,s_goal), it's a single model now
        :param horizon:
        """

        super().__init__()

        self.simulator = simulator
        self.sim_policy = sim_policy
        self.D = D
        self.emseble_gcp = gcp
        self.horizon = horizon
        self.high = self.simulator.env.action_space.high[0]
        self.obs_dim = self.simulator.env.observation_space.shape[0]

    def set_state(self, state):
        self.simulator.set_sim_state(state)

    def get_action(self, s):
        goal_list = []
        virtual_state = s
        for index in range(self.horizon):

            virtual_action = self.sim_policy.get_means(virtual_state, deterministic=True)
            virtual_state, _, done, _ = self.simulator.step(virtual_action * self.high)
            goal_list.append(virtual_state.tolist())
            if done:
                break

        best_reward = -np.inf
        best_goal = goal_list[0]

        for goal in goal_list:
            current_reward = self.D.get_reward(s, goal)
            if current_reward > best_reward:
                best_goal = goal
                best_reward = current_reward
            else:
                pass

        a, std = self.emseble_gcp.get_action(s, best_goal)

        return a
