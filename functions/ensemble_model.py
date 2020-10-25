'''
  A wrapper class for ensemble models.
  @python version : 3.6.4
'''

import numpy as np
from functions.quick_load import goal_conditioned_policy


class model_army():
    def __init__(self, model_num: int = 5, model_path="./model_path"):
        self.model_num = model_num
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.model_list = []

        for i in range(self.model_num):
            self.model_list.append(goal_conditioned_policy(model_path=self.model_path, model_name=str(i)))

        self.obs_dim = self.model_list[-1].obs_dim

    def get_action(self, state, goal):
        import random

        action_list = [gcp.get_action(state, goal) for gcp in self.model_list]

        vstack_al = np.vstack(action_list)
        std = vstack_al.std(axis=0).sum()

        action = np.array(random.choice(action_list))

        return action, std

    def get_std(self, state, goal):
        action_list = [gcp.get_action(state, goal) for gcp in self.model_list]

        vstack_al = np.vstack(action_list)
        std = vstack_al.std(axis=0).mean()

        return std

    def get_batch_std(self, state, state_):
        state = np.array(state).reshape([-1, self.obs_dim])
        state_ = np.array(state_).reshape([-1, self.obs_dim])

        action_batch = np.array([gcp.get_batch_action(state, state_) for gcp in self.model_list])
        std_batch = action_batch.std(axis=0).mean(axis=-1).squeeze()

        return std_batch
