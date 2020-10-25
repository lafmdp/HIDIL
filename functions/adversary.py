'''
  Weighted discriminator for training policy in misspecified simulator.
  @python version : 3.6.8
'''

import os
import numpy as np
import tensorflow as tf
from utils.utils import hype_parameters


def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)


def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class WeightedTransitionClassifier(object):

    def __init__(self, obs_dim=1, model_path=None, have_model=False, model_name='discriminator'):

        self.model_path = model_path

        self.graph = tf.Graph()
        self.obs_dim = obs_dim

        self.learning_rate = 1e-5

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.graph, config=config)

        self.model_name = model_name

        with self.sess.as_default(), self.graph.as_default():
            self.build_network()
            self.def_saver()

            if have_model:
                self.load_model()
            else:
                self.sess.run(tf.global_variables_initializer())

    def def_saver(self):
        with self.sess.as_default(), self.graph.as_default():
            self.all_saver = tf.train.Saver(tf.global_variables())

    def build_network(self):

        with tf.variable_scope('discriminator'):
            self.real_state = tf.placeholder(tf.float32, [None, self.obs_dim], name='state')
            self.real_state_ = tf.placeholder(tf.float32, [None, self.obs_dim], name='next_state')

            self.fake_state = tf.placeholder(tf.float32, [None, self.obs_dim], name='fake_state')
            self.fake_state_ = tf.placeholder(tf.float32, [None, self.obs_dim], name='fake_next_state')

            real_data = tf.concat((self.real_state, self.real_state_), axis=-1)
            fake_data = tf.concat((self.fake_state, self.fake_state_), axis=-1)

            self.d_fake_logit, self.d_fake_prob = self.discriminator(fake_data, reuse=False)
            self.d_real_logit, self.d_real_prob = self.discriminator(real_data, reuse=True)

            self.reward_op = -tf.log(1 - self.d_fake_prob + 1e-8)

        with tf.variable_scope("gan_loss_train"):
            self.std_weight = tf.placeholder(tf.float32, [None, 1], name='fake_next_state')

            # discriminator loss
            # fake loss will be weighted by ensemble model's std
            self.weight = tf.nn.softmax(-self.std_weight, axis=0)
            fake_loss = tf.reduce_sum(self.weight * tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logit,
                                                                                            labels=tf.zeros_like(
                                                                                                self.d_fake_logit)))
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_logit,
                                                                               labels=tf.ones_like(self.d_real_logit)))

            d_loss = fake_loss + real_loss

            logits = tf.concat([self.d_fake_logit, self.d_real_logit], 0)
            entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
            entropy_loss = -0.001 * entropy

            self.discriminator_loss = d_loss + entropy_loss

            self.train_dis = tf.train.AdamOptimizer(learning_rate=hype_parameters["d_lr"]).minimize(
                self.discriminator_loss)

            self.fake_accuracy = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(self.d_fake_logit) < 0.5))
            self.real_accuracy = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(self.d_real_logit) > 0.5))

    def discriminator(self, ipt, reuse):

        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            out = tf.layers.dense(inputs=ipt, units=100, activation=tf.nn.relu, name='layer1')
            out = tf.layers.dense(inputs=out, units=100, activation=tf.nn.relu, name='layer2')
            logit = tf.layers.dense(inputs=out, units=1, activation=None, name='out')

            prob = tf.nn.sigmoid(logit)

        return logit, prob

    def train(self, expert_samples, fake_batch):
        s = np.vstack(fake_batch["gail_state"]).astype(np.float32)
        s_ = np.vstack(fake_batch["gail_state_"]).astype(np.float32)
        fake_std_weight = np.hstack(fake_batch["std_weight"]).astype(np.float32).reshape([-1, 1])

        fake_stack = np.hstack((s, s_))

        this_batch = {
            "state": s,
            "state_": s_
        }

        return self.train_epoch(expert_samples.sample(fake_stack.shape[0]), this_batch, fake_std_weight)

    def train_epoch(self, real_batch, fake_batch, std_weight):
        with self.sess.as_default(), self.sess.graph.as_default():
            d_loss, fa, ra, _ = self.sess.run(
                [self.discriminator_loss, self.fake_accuracy, self.real_accuracy, self.train_dis],
                feed_dict={self.real_state: real_batch["state"],
                           self.real_state_: real_batch["state_"],
                           self.fake_state: fake_batch["state"],
                           self.fake_state_: fake_batch["state_"],
                           self.std_weight: std_weight
                           })

        ret = {
            "d_loss": d_loss,
            "fa": fa,
            "ra": ra
        }

        return ret

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

    def save_model(self):

        with self.graph.as_default():
            self.all_saver.save(self.sess, os.path.join(self.model_path, self.model_name))

    def load_model(self):

        with self.graph.as_default():
            self.all_saver.restore(self.sess, save_path=os.path.join(self.model_path, self.model_name))
