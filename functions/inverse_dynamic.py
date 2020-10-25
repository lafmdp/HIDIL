'''
  Generate action from a state pair, id(s,s')->a.
  @python version : 3.6.8
'''

import os
import numpy as np
import tensorflow as tf


class goal_conditioned_policy(object):

    def __init__(self, state_dim=1, act_dim=1, model_path=None, have_model=False, model_name='goal_conditioned_policy'):
        self.model_path = model_path
        self.model_name = model_name
        self.graph = tf.Graph()
        self.stack_dim = state_dim * 2
        self.act_dim = act_dim

        self.learning_rate = 1e-3
        self.global_step = 1

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.graph, config=config)

        self.build_network()
        self.def_saver()

        with self.sess.as_default(), self.graph.as_default():

            if have_model:
                self.load_model()
            else:
                self.sess.run(tf.global_variables_initializer())

    def def_saver(self):
        with self.sess.as_default(), self.graph.as_default():
            self.all_saver = tf.train.Saver(tf.global_variables())

    def build_network(self):

        with self.sess.as_default(), self.graph.as_default():
            self.stack_state = tf.placeholder(tf.float32, [None, self.stack_dim], name='stack_state')
            self.a_label = tf.placeholder(tf.float32, [None, self.act_dim], name='real_action')

            with tf.name_scope('id_network'):
                out = self.stack_state

                out = tf.layers.dense(out, 100, activation=tf.nn.relu)
                out = tf.layers.dense(out, 100, activation=tf.nn.relu)

                self.action = tf.layers.dense(out, self.act_dim, activation=None)

            with tf.name_scope("train"):
                self.mse_loss = tf.losses.mean_squared_error(self.a_label, self.action)
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.mse_loss)

    def train(self, batch):

        with self.sess.as_default(), self.sess.graph.as_default():
            ipt = np.hstack((batch["state"], batch["state_"]))

            loss, _ = self.sess.run([self.mse_loss, self.train_op],
                                    feed_dict={self.stack_state: ipt,
                                               self.a_label: batch["action"]})

        return loss

    def get_loss(self, batch):

        with self.sess.as_default(), self.sess.graph.as_default():
            ipt = np.hstack((batch["state"], batch["state_"]))

            loss = self.sess.run(self.mse_loss, feed_dict={self.stack_state: ipt,
                                                           self.a_label: batch["action"]})

        return loss

    def get_action(self, stack_state, state_):

        stack_state = np.array(stack_state).reshape([1, -1])
        state_ = np.array(state_).reshape([1, -1])
        ipt = np.hstack((stack_state, state_))

        action = self.sess.run(self.action, feed_dict={
            self.stack_state: ipt,
        })[0]

        return action

    def save_model(self):

        with self.graph.as_default():
            self.all_saver.save(self.sess, os.path.join(self.model_path, self.model_name))

    def load_model(self):

        with self.graph.as_default():
            self.all_saver.restore(self.sess, save_path=os.path.join(self.model_path, self.model_name))
