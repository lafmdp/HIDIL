'''
  Proximal policy optimization method for policy.
  @python version : 3.6.4
'''

import os, datetime
import numpy as np
import tensorflow as tf
from utils.utils import hype_parameters


nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class policy():

    def __init__(self,
                 state_space=11,
                 action_space=6,
                 have_model=False,
                 model_name='policy',
                 model_path='./Documents/PolicyModel/{}/'.format(nowTime),
                 ):

        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.state_space = state_space
        self.action_space = action_space
        self.model_path = model_path
        self.model_name = model_name
        self.have_model = have_model

        self.lamda = hype_parameters["lamda"]
        self.gamma = hype_parameters["gamma"]
        self.batch_size = hype_parameters["batch_size"] if "expert" not in self.model_path else 5000
        self.epoch_num = hype_parameters["epoch_num"]
        self.clip_value = hype_parameters["clip_value"]
        self.c_1 = hype_parameters["c_1"]
        self.c_2 = hype_parameters["c_2"]
        self.init_lr = hype_parameters["init_lr"]
        self.lr_epsilon = hype_parameters["lr_epsilon"]

        self.build_graph()

        self.n_training = 0

        with self.sess.as_default(), self.graph.as_default():

            self.saver = tf.train.Saver(self.get_variables())

            if have_model:
                self.load_model()
            else:
                self.sess.run(tf.global_variables_initializer())

    def build_graph(self):
        with self.sess.as_default(), self.graph.as_default():
            with tf.variable_scope('policy'):
                self.obs = tf.placeholder(dtype=tf.float32, shape=[None, self.state_space], name='obs')
                self.global_step = tf.Variable(0, trainable=False)
                self.learning_rate = tf.train.noisy_linear_cosine_decay(
                    learning_rate=self.init_lr, decay_steps=100000, global_step=self.global_step,
                    initial_variance=0.01, variance_decay=0.1, num_periods=0.2, alpha=0.05, beta=0.2)
                self.add_global = self.global_step.assign_add(1)

                with tf.variable_scope("Net"):
                    with tf.variable_scope('action'):
                        out = tf.layers.dense(self.obs, 128, tf.nn.relu)
                        out = tf.layers.dense(out, 256, tf.nn.relu)
                        out = tf.layers.dense(out, 128, tf.nn.relu)
                        out = tf.layers.dense(out, 64, tf.nn.relu)

                        self.means = tf.layers.dense(out, self.action_space, tf.nn.tanh, name="means")

                        self.log_vars = tf.constant(-2, dtype=tf.float32, shape=[self.action_space, ],
                                                    name='log_variance')

                        sampled_act = (self.means +
                                       tf.exp(self.log_vars / 2.0) *
                                       tf.truncated_normal(shape=(self.action_space,)))

                        self.sampled_act = tf.clip_by_value(sampled_act, -1, 1)

                    with tf.variable_scope('value'):
                        out2 = tf.layers.dense(inputs=self.obs, units=128, activation=tf.nn.relu)
                        out2 = tf.layers.dense(inputs=out2, units=256, activation=tf.nn.relu)
                        out2 = tf.layers.dense(inputs=out2, units=128, activation=tf.nn.relu)
                        self.value = tf.layers.dense(inputs=out2, units=1, activation=None)

                self.scope = tf.get_variable_scope().name

            with tf.name_scope("policy_train"):
                with tf.name_scope('train_input'):
                    self.returns = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward_to_go')
                    self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
                    self.old_actions = tf.placeholder(tf.float32, [None, self.action_space], name='action_done')
                    self.old_means = tf.placeholder(tf.float32, [None, self.action_space], name='old_means')
                    self.first_step_return = tf.placeholder(tf.float32, [None], name="first_return")
                    self.trajectory_len = tf.placeholder(tf.float32, [None], name="traj_len")
                    self.batch_lr = tf.placeholder(tf.float32, [None], name="batch_learning_rate")
                    self.batch_actor_loss = tf.placeholder(tf.float32, [None], name="batch_a_loss")
                    self.batch_critic_loss = tf.placeholder(tf.float32, [None], name="batch_c_loss")
                    self.batch_entropy = tf.placeholder(tf.float32, [None], name="batch_entropy")

                with tf.name_scope('loss_and_train'):
                    with tf.name_scope('policy_loss'):
                        self._logprob()
                        ratios = tf.exp(self.logp - self.logp_old)
                        clipped_ratios = tf.clip_by_value(ratios,
                                                          clip_value_min=1 - self.clip_value,
                                                          clip_value_max=1 + self.clip_value,
                                                          name='continuous_ratios')
                        loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                        self.actor_loss = -tf.reduce_mean(loss_clip)

                    with tf.name_scope('value_loss'):
                        self.critic_loss = tf.losses.mean_squared_error(self.returns, self.value)

                    with tf.name_scope('entropy_loss'):
                        self.entropy = tf.reduce_sum(
                            0.5 * self.log_vars + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
                        self.entropy_loss = -self.entropy

                    with tf.name_scope('total_loss'):
                        total_loss = self.actor_loss + self.c_1 * self.critic_loss  # + self.c_2 * self.entropy_loss

                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.lr_epsilon)

                    self.train_op = optimizer.minimize(total_loss)

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions
        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.old_actions - self.means) /
                                     tf.exp(self.log_vars), axis=-1)

        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.log_vars)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.old_actions - self.old_means) /
                                         tf.exp(self.log_vars), axis=-1)
        self.logp_old = logp_old

    def get_action(self, obs):
        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            actions, value = \
                self.sess.run([self.sampled_act, self.value], feed_dict={self.obs: obs})

            ret = {
                "actions": actions[0],
                "value": value[0]
            }

        return ret

    def get_means(self, obs):

        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            actions = self.sess.run(self.means, feed_dict={self.obs: obs})

        return actions[0]

    def get_value(self, obs):

        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            value = self.sess.run(self.value, feed_dict={self.obs: obs})

        return value[0]

    def get_variables(self):
        with self.sess.as_default(), self.graph.as_default():
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def train(self, batch):
        with self.sess.as_default(), self.graph.as_default():

            # convert list to numpy array
            state = np.vstack(batch["state"]).astype(np.float32).squeeze()
            old_action = np.vstack(batch["action"]).astype(dtype=np.float32).squeeze()
            gae = np.vstack(batch["gae"]).astype(np.float32).squeeze()
            ret = np.hstack(batch["return"]).astype(dtype=np.float32).squeeze()

            old_mean = self.sess.run(self.means, feed_dict={self.obs: state})
            old_mean = np.array(old_mean, dtype=np.float32).squeeze()

            s_s = self.state_space
            a_s = self.action_space

            gae = gae[:, np.newaxis]
            ret = ret[:, np.newaxis]

            if a_s == 1:
                old_action = old_action[:, np.newaxis]
                old_mean = old_mean[:, np.newaxis]

            actor_loss = []
            critic_loss = []
            entropy = []
            learning_r = []

            dataset = np.hstack((state, old_action, old_mean, gae, ret))
            np.random.shuffle(dataset)

            states = dataset[:, :s_s]
            old_actions = dataset[:, s_s:s_s + a_s]
            old_means = dataset[:, s_s + a_s:s_s + 2 * a_s]
            gaes = dataset[:, -2]
            rets = dataset[:, -1]

            gaes = np.squeeze(gaes)
            rets = rets[:, np.newaxis]

            if s_s == 1:
                old_actions = old_actions[:, np.newaxis]

            sample_num = dataset.shape[0]

            for i in range(self.epoch_num):

                start = 0
                end = min(start + self.batch_size, sample_num)

                while start < sample_num:
                    a_loss, c_loss, lr, entropy_loss, _, _ = \
                        self.sess.run([self.actor_loss,
                                       self.critic_loss,
                                       self.learning_rate,
                                       self.entropy_loss,
                                       self.train_op,
                                       self.add_global],
                                      feed_dict={self.obs: states[start:end],
                                                 self.returns: rets[start:end],
                                                 self.gaes: gaes[start:end],
                                                 self.old_actions: old_actions[start:end],
                                                 self.old_means: old_means[start:end]})

                    actor_loss.append(a_loss)
                    critic_loss.append(c_loss)
                    learning_r.append(lr.mean())
                    entropy.append(-entropy_loss)

                    start += self.batch_size
                    end = min(start + self.batch_size, sample_num)

    def save_model(self):

        with self.sess.as_default(), self.graph.as_default():
            self.saver.save(self.sess, save_path=os.path.join(self.model_path, self.model_name))

    def load_model(self):
        with self.sess.as_default(), self.graph.as_default():
            self.saver.restore(self.sess, save_path=os.path.join(self.model_path, self.model_name))
