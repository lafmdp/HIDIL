'''
  Replay buffer for storing samples.
  @python version : 3.6.8
'''

import numpy as np


class Buffer(object):

    def append(self, *args):
        pass

    def sample(self, *args):
        pass


class MS_Buffer(Buffer):

    def __init__(self, s_space=1, a_space=1, window_size=1, size=1000000, batch_size=256):

        self.size = size
        self.batch_size = batch_size

        self.s_space = s_space
        self.a_space = a_space
        self.window_size = window_size

        self.buffer = np.array([], dtype=np.float32)

    def append(self, s, a, s_):

        sample_num = self.buffer.shape[0]
        if sample_num > self.size:
            index = np.random.choice(sample_num, self.size, replace=False).tolist()
            self.buffer = self.buffer[index]
        else:
            pass

        s = np.vstack(s).astype(np.float32)
        a = np.vstack(a).astype(np.float32)
        s_ = np.vstack(s_).astype(np.float32)

        recorder = np.hstack((s, a, s_))

        if sample_num == 0:
            self.buffer = recorder.copy()
        else:
            self.buffer = np.vstack((self.buffer, recorder))

    def sample(self, batchsize=None):
        size = batchsize if batchsize is not None else self.batch_size

        sample_num = self.buffer.shape[0]
        if sample_num < size:
            sample_index = np.random.choice(sample_num, size, replace=True).tolist()
        else:
            sample_index = np.random.choice(sample_num, size, replace=False).tolist()

        sample = self.buffer[sample_index]

        s = sample[:, :self.s_space]
        a = sample[:, self.s_space:self.s_space + self.a_space]
        s_ = sample[:, self.s_space + self.a_space:self.s_space + self.a_space + self.s_space]

        s = np.array(s).astype(np.float32)
        a = np.array(a).astype(np.float32)
        s_ = np.array(s_).astype(np.float32)

        ret = {
            "state": s,
            "action": a,
            "state_": s_
        }

        return ret

    def sample_ss_(self):

        sample_num = self.buffer.shape[0]
        if sample_num < self.batch_size:
            sample_index = np.random.choice(sample_num, self.batch_size, replace=True).tolist()
        else:
            sample_index = np.random.choice(sample_num, self.batch_size, replace=False).tolist()

        sample = self.buffer[sample_index]

        s = sample[:, :self.s_space]
        s_ = sample[:, self.s_space + self.a_space:self.s_space + self.a_space + self.s_space]

        s = np.array(s).astype(np.float32)
        s_ = np.array(s_).astype(np.float32)

        ret = {
            "state": s,
            "state_": s_
        }

        return ret
