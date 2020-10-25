'''
  Load existing TF model quickly.
  @python version : 3.6.4
'''

import numpy as np
import tensorflow as tf


class load_base(object):
    def __init__(self, path='./policy', model_name="policy"):
        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.graph)
        model_path = "{}/{}".format(path, model_name)

        with self.sess.as_default(), self.graph.as_default():
            saver = tf.train.import_meta_graph(model_path + '.meta')
            saver.restore(self.sess, model_path)

            self.specific_tensor()

    def specific_tensor(self):
        pass


class policy(load_base):
    def __init__(self, model_path):

        super().__init__(model_path, "policy")

    def specific_tensor(self):
        self.state = self.graph.get_tensor_by_name("policy/obs:0")
        self.deterministic_action = self.graph.get_tensor_by_name("policy/Net/action/means/Tanh:0")
        self.stochastic_action = self.graph.get_tensor_by_name("policy/Net/action/clip_by_value:0")
        self.value = self.graph.get_tensor_by_name("policy/Net/value/dense_3/BiasAdd:0")

    def get_action(self, obs):
        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            actions, value = \
                self.sess.run([self.stochastic_action, self.value], feed_dict={self.state: obs})

            ret = {
                "actions": actions[0],
                "value": value[0]
            }

        return ret

    def get_means(self, obs, deterministic=True):

        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            actions = self.sess.run(self.deterministic_action if deterministic else
                                    self.stochastic_action, feed_dict={self.state: obs})

        return actions[0]

    def get_value(self, obs):

        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            value = self.sess.run(self.value, feed_dict={self.state: obs})

        return value[0]


class behavior_cloning(load_base):
    def __init__(self, model_path):
        super().__init__(model_path, "behavior_cloning")

    def specific_tensor(self):
        self.state = self.graph.get_tensor_by_name("stack_state:0")
        self.action = self.graph.get_tensor_by_name("bc_network/dense_2/BiasAdd:0")

    def get_action(self, state):
        with self.sess.as_default(), self.sess.graph.as_default():
            state = np.array(state).reshape([1, -1])

            action = self.sess.run(self.action, feed_dict={
                self.state: state,
            })[0]

        return dict(actions=action)

    def get_means(self, state):
        with self.sess.as_default(), self.sess.graph.as_default():
            state = np.array(state).reshape([1, -1])

            action = self.sess.run(self.action, feed_dict={
                self.state: state,
            })[0]

        return action


class TransitionClassifier(load_base):
    def __init__(self, model_path, model_name="discriminator"):
        super().__init__(model_path, model_name)

    def specific_tensor(self):
        self.fake_state = self.graph.get_tensor_by_name("discriminator/fake_state:0")
        self.fake_state_ = self.graph.get_tensor_by_name("discriminator/fake_next_state:0")
        self.reward_op = self.graph.get_tensor_by_name("discriminator/Neg:0")
        self.obs_dim = self.fake_state.get_shape().as_list()[-1]

    def get_reward(self, state, state_):
        state = np.array(state).reshape([1, -1])
        state_ = np.array(state_).reshape([1, -1])

        with self.sess.as_default():
            reward = self.sess.run(self.reward_op, feed_dict={self.fake_state: state,
                                                              self.fake_state_: state_})

        return float(reward[0])

    def get_batch_reward(self, state, state_):
        state = np.array(state).reshape([-1, self.obs_dim])
        state_ = np.array(state_).reshape([-1, self.obs_dim])

        with self.sess.as_default():
            reward = self.sess.run(self.reward_op, feed_dict={self.fake_state: state,
                                                              self.fake_state_: state_})

        return reward


class inverse_dynamic(load_base):
    def __init__(self, model_path):
        super().__init__(model_path, "inverse_dynamic")

    def specific_tensor(self):
        self.state = self.graph.get_tensor_by_name("stack_state:0")
        self.action = self.graph.get_tensor_by_name("id_network/dense_2/BiasAdd:0")

    def get_action(self, stack_state, state_):
        stack_state = np.array(stack_state).reshape([1, -1])
        state_ = np.array(state_).reshape([1, -1])
        ipt = np.hstack((stack_state, state_))

        action = self.sess.run(self.action, feed_dict={
            self.state: ipt,
        })[0]

        return action


class goal_conditioned_policy(load_base):
    def __init__(self, model_path, model_name="goal_conditioned_policy"):
        super().__init__(model_path, model_name)

    def specific_tensor(self):
        self.state = self.graph.get_tensor_by_name("stack_state:0")
        self.action = self.graph.get_tensor_by_name("id_network/dense_2/BiasAdd:0")

        self.obs_dim = int(self.state.get_shape().as_list()[-1] / 2)

    def get_action(self, state, state_):
        state = np.array(state).reshape([1, -1])
        state_ = np.array(state_).reshape([1, -1])
        ipt = np.hstack((state, state_))

        with self.sess.as_default(), self.graph.as_default():
            action = self.sess.run(self.action, feed_dict={
                self.state: ipt
            })[0]

        return action

    def get_batch_action(self, state, state_):
        state = np.array(state).reshape([-1, self.obs_dim])
        state_ = np.array(state_).reshape([-1, self.obs_dim])
        ipt = np.hstack((state, state_))

        with self.sess.as_default(), self.graph.as_default():
            action = self.sess.run(self.action, feed_dict={
                self.state: ipt
            })

        return action
