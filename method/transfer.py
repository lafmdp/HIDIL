"""
  Main loop file.
  @python version : 3.6.8
"""

from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process, Manager
import gym
from utils.utils import *
from method import method
from functions import quick_load
from functions.policy import policy
from utils.replay_buffer import MS_Buffer
from functions.ensemble_model import model_army
from functions.adversary import WeightedTransitionClassifier
from functions.inverse_dynamic import goal_conditioned_policy


plt.switch_backend('agg')  # Run without display

max = lambda a, b: a if a > b else b


class transfer():

    def __init__(self, parameters):

        self.env_name = parameters.env

        env = gym.make(self.env_name)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.high = env.action_space.high[0]
        self.random_seed = 0

        print("obs_space:", self.obs_dim)
        print("act_space:", self.act_dim)
        print("act_high:", env.action_space.high)
        print("act_low:", env.action_space.low)

        env.close()
        self.rl_keys = ["state", "state_", "action", "reward", "std_weight", "mujoco_reward",
                        "gae", "return", "sum_reward", "trajectory_len", "gail_state", "gail_state_"]

        # keys for RL training
        self.expert_data_keys = ["state", "action", "state_",
                                 "ms_state", "ms_action", "ms_state_"]

        # keys for store summary
        self.summary_keys = ["d_loss", "fa", "ra", "d_reward", "mujoco_reward",
                             "std", "scores", "expert_dis", "our_dis"]

        self.args = parameters
        self.process_num = parameters.process_num

        self.data_path = check_path('./documents/{}/data'.format(self.args.log_index))

    def reset(self):

        self.policy_path = check_path('./documents/{}/policy'.format(self.args.log_index))
        self.expert_path = './expert_model/{}/{}/{}'.format(self.transfer_type, self.env_name, self.variety_degree)
        self.discriminator_path = check_path('./documents/{}/gan_learning'.format(self.args.log_index))
        self.gcp_path = check_path('./documents/{}/goal_conditioned_policy'.format(self.args.log_index))
        generate_log(self.args)

        self.random_seed = self.args.random_seed

        # Expert's samples are stored here
        self.gcp_expert_target_samples = MS_Buffer(s_space=self.obs_dim,
                                                   a_space=self.act_dim,
                                                   window_size=args.horizon,
                                                   batch_size=args.batch_size)

        p = Process(target=self.initNetworks)
        p.start()
        p.join()

        self.summary = {}
        for key in self.summary_keys:
            self.summary[key] = []

    def set_training_configuration(self, env_name, variety_degree, transfer_type):
        self.env_name = env_name
        self.variety_degree = variety_degree
        self.transfer_type = transfer_type

    def policyWorker(self, points_num, share_lock):

        pi = quick_load.policy(self.policy_path)
        discriminator = quick_load.TransitionClassifier(model_path=self.discriminator_path)
        ensemble_model = model_army(model_num=self.args.ensemble_num, model_path=self.gcp_path)
        env = self.simulator_env

        batch = {}
        for key in self.rl_keys:
            batch[key] = []

        all_step_nums = self.args.points_num
        import random
        env.seed(random.randint(0, 100000))

        while True:

            s = env.reset()

            traj_batch = {
                "state": [],
                "state_": [],
                "actions": [],
                "reward": [],
                "value": []
            }

            vanilla_reward = 0

            while True:

                if points_num.value > all_step_nums:
                    return batch

                ret = pi.get_action(s)

                s_, r, done, _ = env.step(ret['actions'] * self.high)
                vanilla_reward += r

                r = 0  # It's invisible to MuJoCo reward

                traj_batch["state"].append(s)
                traj_batch["state_"].append(s_)
                traj_batch["reward"].append(r)
                traj_batch["actions"].append(ret["actions"])
                traj_batch["value"].append(ret["value"])

                s = s_

                if done:

                    share_lock.acquire()
                    points_num.value += len(traj_batch["state"])
                    share_lock.release()

                    # modify state for horizon-adaptive goal conditioned policy
                    new_reward_list = np.zeros_like(np.array(traj_batch["reward"]), dtype=np.float32)
                    traj_len = len(traj_batch["state"])

                    state_list = []
                    state__list = []

                    for index in range(traj_len):
                        for j in range(self.args.horizon):
                            if index + j < traj_len:
                                state_ = traj_batch["state_"][index + j].copy().tolist()
                            else:
                                state_ = traj_batch["state_"][-1].copy().tolist()

                            state_list.append(traj_batch["state"][index])
                            state__list.append(state_)

                            batch["gail_state"].append(traj_batch["state"][index])
                            batch["gail_state_"].append(state_)

                    D_reward = discriminator.get_batch_reward(state_list, state__list)
                    std_weight = ensemble_model.get_batch_std(state_list, state__list)
                    batch["std_weight"].append(std_weight)

                    for i in range(traj_len):
                        new_reward_list[i] = D_reward[
                                             (i * self.args.horizon):(i * self.args.horizon + self.args.horizon)].max()

                    traj_batch["reward"] = new_reward_list.tolist()

                    batch["sum_reward"].append(sum(traj_batch["reward"]))

                    v = pi.get_value(s)
                    real_next = traj_batch["value"][1:] + [np.array(v)]
                    ret = get_return(traj_batch["reward"])
                    gae = get_gaes(traj_batch["reward"], traj_batch["value"], real_next)

                    batch["state"].append(traj_batch["state"])
                    batch["state_"].append(traj_batch["state_"])
                    batch["action"].append(traj_batch["actions"])
                    batch["gae"].append(gae)
                    batch["return"].append(ret)
                    batch["mujoco_reward"].append(vanilla_reward)
                    batch["trajectory_len"].append(len(traj_batch["state"]))
                    batch["reward"].append(traj_batch["reward"])

                    break

    def train_pi_worker(self, policy_batch):

        fa, ra, loss = 0, 0, 0  # fake samples accuracy/ real samples accuracy/ loss

        pi = policy(have_model=True, action_space=self.act_dim, state_space=self.obs_dim,
                    model_path=self.policy_path)

        dis = WeightedTransitionClassifier(obs_dim=self.obs_dim,
                                           model_path=self.discriminator_path,
                                           have_model=True)

        gail_epoch = self.args.gail_epoch if self.iter > 10 else 10

        for _ in range(gail_epoch):
            ret = dis.train(self.gcp_expert_target_samples, policy_batch.copy())

        fa = ret["fa"]
        ra = ret["ra"]
        loss = ret["d_loss"]

        dis.save_model()

        pi.train(policy_batch)
        pi.save_model()

        return fa, ra, loss

    def train_policy(self):

        p = Pool(self.process_num)

        policy_batch = {}

        for key in self.rl_keys:
            policy_batch[key] = []

        points_num = Manager().Value("l", 0)
        share_lock = Manager().Lock()

        results = []

        for i in range(self.process_num):
            results.append(p.apply_async(self.policyWorker, args=(points_num, share_lock,)))

        p.close()
        p.join()

        for res in results:

            res = res.get()
            for key in self.rl_keys:
                policy_batch[key] += res[key]

        p = Pool(1)
        results = []

        for i in range(1):
            results.append(p.apply_async(self.train_pi_worker, args=(policy_batch,)))

        p.close()
        p.join()

        for res in results:
            fa, ra, loss = res.get()

        self.summary["mujoco_reward"].append(np.array(policy_batch["mujoco_reward"]).mean())
        self.summary["d_reward"].append(np.array(policy_batch["sum_reward"]).mean())
        self.summary["d_loss"].append(loss)
        self.summary["fa"].append(fa)
        self.summary["ra"].append(ra)
        self.summary["std"].append(np.hstack(policy_batch["std_weight"]).astype(np.float32).reshape([-1]).mean())

    def initNetworks(self):

        cur_model = WeightedTransitionClassifier(obs_dim=self.obs_dim,
                                                 model_path=self.discriminator_path,
                                                 have_model=False)

        cur_model.save_model()

        pi = policy(have_model=False,
                    action_space=self.act_dim,
                    state_space=self.obs_dim,
                    model_path=self.policy_path)

        pi.save_model()

        for i in range(self.args.ensemble_num):
            idm_ours = goal_conditioned_policy(state_dim=self.obs_dim,
                                               act_dim=self.act_dim,
                                               model_path=self.gcp_path,
                                               model_name=str(i),
                                               have_model=False)
            idm_ours.save_model()
            del idm_ours

    def evaluateWorker(self, eva_num, share_lock):

        env = self.target_env

        ensemble_bc = model_army(model_num=self.args.ensemble_num,
                                 model_path=self.gcp_path)
        pi = quick_load.policy(self.policy_path)
        dis = quick_load.TransitionClassifier(model_path=self.discriminator_path)

        p = method.ours(self.simulator, pi, ensemble_bc, dis, self.args.horizon)

        batch = {
            "sum_reward": []
        }

        import random
        env.seed(random.randint(0, 100000))

        game_num = 0

        while True:
            game_num += 1

            s = env.reset()

            sum_reward = 0

            while True:
                if eva_num.value > self.args.eva_num:
                    return batch

                p.set_state(env.sim.get_state())
                a = p.get_action(s)

                s_, r, done, _ = env.step(a * self.high)

                sum_reward += r
                s = s_

                if done:
                    share_lock.acquire()
                    eva_num.value += 1
                    share_lock.release()
                    batch["sum_reward"].append(sum_reward)

                    break

    def evaluate(self, eva_type):

        p = Pool(self.process_num)

        batch = {
            "sum_reward": []
        }

        eva_num = Manager().Value("l", 0)
        share_lock = Manager().Lock()

        results = []

        for i in range(self.process_num):
            results.append(p.apply_async(self.evaluateWorker, args=(eva_num, share_lock,)))

        p.close()
        p.join()

        for res in results:
            res = res.get()
            batch["sum_reward"] += res["sum_reward"]

        rt = np.hstack(batch["sum_reward"]).astype(dtype=np.float32).squeeze()

        log = ('Evaluation of our methods: mean:', rt.mean(), 'max:', rt.max(), 'min:', rt.min())
        generate_log(log)

        return rt.mean()

    def sampleEnv(self, points_num, share_lock):

        pi = quick_load.policy(self.expert_path)

        env = self.target_env

        batch = {}
        for key in self.expert_data_keys:
            batch[key] = []

        import random
        env.seed(random.randint(0, 100000))

        game_num = 0
        lower_bound = 2500

        while True:

            s = env.reset()

            traj_batch = {
                "state": [],
                "action": [],
                "state_": []
            }
            sum_reward = 0

            while True:
                if points_num.value >= self.args.expert_demonstration_traj:
                    return batch

                means = pi.get_means(s, deterministic=False)

                s_, r, done, _ = env.step(means * self.high)
                sum_reward += r
                traj_batch["state"].append(s)
                traj_batch["action"].append(means)
                traj_batch["state_"].append(s_)

                s = s_

                if done:
                    game_num += 1

                    if game_num > self.args.expert_demonstration_traj:
                        game_num = 0
                        lower_bound -= 100

                    # Bad samples are ought to be abandoned.
                    if sum_reward < lower_bound:
                        break
                    else:
                        share_lock.acquire()
                        points_num.value += 1
                        share_lock.release()

                        # multi-step data
                        traj_len = len(traj_batch["state"])
                        for index in range(traj_len):
                            for i in range(self.args.horizon):
                                if index + i >= traj_len:
                                    break

                                batch["ms_state"].append(traj_batch["state"][index])
                                batch["ms_action"].append(traj_batch["action"][index])
                                state_ = traj_batch["state_"][index + i].copy().tolist()
                                batch["ms_state_"].append(state_)

                        # one step data
                        batch["state"].append(traj_batch["state"])
                        batch["action"].append(traj_batch["action"])
                        batch["state_"].append(traj_batch["state_"])

                        break

    def trainIdWorker(self):
        """
        Train inverse dynamic model for expert on the target.
        """

        for i in range(self.args.ensemble_num):
            idt = goal_conditioned_policy(model_path=self.gcp_path,
                                          have_model=True,
                                          state_dim=self.obs_dim,
                                          model_name=str(i),
                                          act_dim=self.act_dim)

            for _ in range(self.args.id_epoch):
                idt.train(self.gcp_expert_target_samples.sample())

            idt.save_model()
            del idt

    def trainId(self):

        p = Pool(self.process_num)

        batch = {}
        for key in self.expert_data_keys:
            batch[key] = []

        points_num = Manager().Value("l", 0)
        share_lock = Manager().Lock()
        results = []

        for i in range(self.process_num):
            results.append(p.apply_async(self.sampleEnv, args=(points_num, share_lock,)))

        p.close()
        p.join()

        for res in results:
            res = res.get()
            for key in self.expert_data_keys:
                batch[key] += res[key]

        self.gcp_expert_target_samples.append(batch["ms_state"], batch["ms_action"], batch["ms_state_"])

        pro = Process(target=self.trainIdWorker)

        pro.start()
        pro.join()

    def collect_distribution(self, points_num, share_lock, policy_type):
        if policy_type == "ours":
            pi = quick_load.policy(self.policy_path)
            env = self.simulator_env
        else:
            pi = quick_load.policy(self.expert_path)
            env = self.target_env

        batch = []

        import random
        env.seed(random.randint(0, 100000))

        while True:

            s = env.reset()

            while True:
                if points_num.value >= self.args.distribution_traj_num:
                    return batch

                means = pi.get_means(s, deterministic=True)

                s_, r, done, _ = env.step(means * self.high)
                batch.append(s)

                s = s_

                if done:
                    share_lock.acquire()
                    points_num.value += 1
                    share_lock.release()

                    break

    def compute_distribution(self):

        p = Pool(self.process_num)

        batch = []

        points_num = Manager().Value("l", 0)
        share_lock = Manager().Lock()
        results = []

        for i in range(self.process_num):
            results.append(p.apply_async(self.collect_distribution, args=(points_num, share_lock, "ours",)))

        p.close()
        p.join()

        for res in results:
            batch += res.get()

        self.summary["our_dis"] = batch

        p = Pool(self.process_num)

        batch = []

        points_num = Manager().Value("l", 0)
        share_lock = Manager().Lock()
        results = []

        for i in range(self.process_num):
            results.append(p.apply_async(self.collect_distribution, args=(points_num, share_lock, "expert",)))

        p.close()
        p.join()

        for res in results:
            batch += res.get()
        self.summary["expert_dis"] = batch

    def run(self, source_env, target_env, simulator):
        """
        MainLoop for one configuration
        :param source_env: Simulation env
        :param target_env: Real world
        :param simulator: Misspecified simulator
        """

        self.simulator_env = source_env
        self.target_env = target_env
        self.simulator = simulator

        self.reset()

        self.trainId()

        self.iter = 0

        tqdm_bar = tqdm(range(self.args.iteration_num))

        for i in tqdm_bar:

            if self.iter % self.args.log_interval == 0:
                reward = self.evaluate("ours")
                self.summary["scores"].append(reward)

            tqdm_bar.set_description('Iterations {} (env:{}, variety:{}, seed:{})'.format(
                i, self.env_name, self.args.variety_degree, self.random_seed
            ))

            self.train_policy()

            self.iter += 1

        self.compute_distribution()

        np.save("./result/summary/{}-{}-{}-{}.summary".format(self.transfer_type,
                                                              self.env_name,
                                                              self.random_seed,
                                                              self.variety_degree), self.summary)
