'''
  Utils functions and some configs.
  @python version : 3.6.8
'''

import os, re, copy, time, random, datetime, argparse
import numpy as np
import tensorflow as tf


nowTime = datetime.datetime.now().strftime('%y-%m-%d%H:%M:%S')
parser = argparse.ArgumentParser(description="Process running arguments")

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

# hype parameter for PPO training
hype_parameters = {
    "gamma": 0.99,
    "lamda": 0.95,
    "batch_size": 512,
    "epoch_num": 10,
    "clip_value": 0.2,
    "c_1": 3,
    "c_2": 0.001,
    "init_lr": 3e-4,
    "d_lr": 3e-4,
    "lr_epsilon": 1e-6
}

# Algorithms running configuration parameters. Add argument if needed.
parser.add_argument('--process_num',
                    default=30, type=int,
                    help='Nums of processes for parallel sampling.')
parser.add_argument('--vg',
                    default="1", type=str,
                    help='Visible gpus.')
parser.add_argument('--discription',
                    default="use ensemble model to provide std weight", type=str,
                    help='Extra discription added to log file.')
parser.add_argument('--env_list',
                    default="Walker2d-v2", type=str,
                    help='Avaliable env, seperated by \';\'.')
parser.add_argument('--batch_size',
                    default=256, type=int,
                    help='default batch size for discriminator.')
parser.add_argument('--points_num',
                    default=10000, type=int,
                    help='Sample Nums per iteration(in state-action pairs).')
parser.add_argument('--eva_num',
                    default=50, type=int,
                    help='Trajectories Nums executed per evaluation.')
parser.add_argument('--iteration_num',
                    default=400, type=int,
                    help='Nums of training iterations for each type of modification.')
parser.add_argument('--id_epoch',
                    default=20000, type=int,
                    help='Nums of gradient step for training a inverse dynamic model.')
parser.add_argument('--expert_demonstration_traj',
                    default=1, type=int,
                    help='Nums of expert demonstrations(in trajectories).')
parser.add_argument('--distribution_traj_num',
                    default=50, type=int,
                    help='Nums of trajectories for estimating distribution(in trajectories).')
parser.add_argument('--log_index',
                    default=nowTime, type=str,
                    help='Current system time for creating files.')
parser.add_argument('--variety_list',
                    default="0.5;1.5;2.0", type=str,
                    help="Customize env mismatch degree as you like."
                         "If you want to serially run multiple tasks, separate them by \';\'.")
parser.add_argument('--ensemble_num',
                    default=5, type=int,
                    help="Nums of ensemble models.")
parser.add_argument('--horizon',
                    default=5, type=int,
                    help="Horizon for local matching.")
parser.add_argument('--transfer_type',
                    default="gravity", type=str,
                    help="Customize env mismatch type as you like."
                         "If you want to serially run multiple tasks, separate them by \';\'")
parser.add_argument('--gail_epoch',
                    default=5, type=int,
                    help="Nums of discriminator gradient steps per policy's gradient step.")
parser.add_argument('--log_interval',
                    default=5, type=int,
                    help="Interval of evaluating and recording data.")
parser.add_argument('--random_seed',
                    type=int,
                    default=10,
                    help="10 is Leo Messi's uniform number.")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.vg


def generate_xml_path():
    import gym, os
    xml_path = os.path.join(gym.__file__[:-11], 'envs/mujoco/assets')

    assert os.path.exists(xml_path)

    return xml_path


gym_xml_path = generate_xml_path()


def record_data(file, content):
    with open(file, 'a+') as f:
        f.write('{}\n'.format(content))


def check_path(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except FileExistsError:
        pass

    return path


def update_xml(index, env_name):
    xml_name = parse_xml_name(env_name)
    os.system('cp ./xml_path/{0}/{1} {2}/{1}}'.format(index, xml_name, gym_xml_path))

    time.sleep(0.2)


def parse_xml_name(env_name):
    if 'walker' in env_name.lower():
        xml_name = "walker2d.xml"
    elif 'hopper' in env_name.lower():
        xml_name = "hopper.xml"
    elif 'halfcheetah' in env_name.lower():
        xml_name = "half_cheetah.xml"
    elif "ant" in env_name.lower():
        xml_name = "ant.xml"
    else:
        raise RuntimeError("No available environment named \'%s\'" % env_name)

    return xml_name


def update_source_env(env_name):
    xml_name = parse_xml_name(env_name)

    os.system(
        'cp ./xml_path/source_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_gravity(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "gravity" in line:
                pattern = re.compile(r"gravity=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "gravity=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_density(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "density" in line:
                pattern = re.compile(r'(?<=density=")\d+\.?\d*')
                a = pattern.findall(line)
                current_num = float(a[0])
                replace_num = current_num * variety_degree
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def update_target_env_friction(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "friction" in line:
                pattern = re.compile(r"friction=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "friction=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def generate_log(extra=None):
    print(extra)
    record_data('./documents/{}/data/log.txt'.format(args.log_index), "{}".format(extra))


def get_gaes(rewards, v_preds, v_preds_next):
    deltas = [r_t + hype_parameters["gamma"] * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + hype_parameters["gamma"] * hype_parameters["lamda"] * gaes[t + 1]

    return gaes


def get_return(rewards):
    dis_rewards = np.zeros_like(rewards).astype(np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * hype_parameters["gamma"] + rewards[t]
        dis_rewards[t] = running_add

    return dis_rewards


def set_global_seeds(i):
    myseed = i  # + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except Exception as e:
        print("Check your tensorflow version")
        raise e
    np.random.seed(myseed)
    random.seed(myseed)


set_global_seeds(args.random_seed)


def check_file_path():
    check_path("./documents")
    check_path("./result")
    check_path("./result/summary")
    check_path("./documents/%s" % args.log_index)
