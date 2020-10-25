'''
  Tmux config file generator.
  @python version : 3.6.8
'''

import yaml
import argparse


# python generate_tmux_yaml.py --num_seeds 4 --env_names "Hopper-v2"
parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--num_seeds',
                    type=int, default=3,
                    help='Nums of random seeds.')
parser.add_argument('--process_num',
                    type=int, default=5,
                    help='Nums of process per panel.')
parser.add_argument('--env_list',
                    default="HalfCheetah-v2;Walker2d-v2;Ant-v2", type=str,
                    help='Environment name separated by \';\' ')
parser.add_argument('--transfer_type',
                    default="gravity", type=str,
                    help='Transfer type.')
parser.add_argument('--conda_name',
                    default="your_venv", type=str,
                    help='Conda environments name.')
parser.add_argument("--traj_num",
                    default=10, type=int,
                    help='Trajectories Nums of expert demonstrations.')
parser.add_argument("--iteration_num",
                    default=300, type=int,
                    help='Nums of training iterations for each type of modification.')
parser.add_argument("--variety_list",
                    default="\"0.5;1.5;2.0\"", type=str,
                    help="Customize env mismatch degree as you like."
                         "If you want to serially run multiple tasks, separate them by \';\'.")

args = parser.parse_args()

run_template = "python main.py " \
               "--transfer_type {} " \
               "--env_list {} " \
               "--process_num {} " \
               "--random_seed {} " \
               "--log_index {} " \
               "--expert_demonstration_traj {} " \
               "--iteration_num {} " \
               "--variety_list {}"

first = True
template = run_template
dir_name = 0
sleep_inverval = 3

config = {"session_name": "transfer-{}".format(args.transfer_type), "windows": []}
env_id = args.env_list
for env_name in env_id.split(';'):

    sleep_index = 0

    for transfer_type in args.transfer_type.split(";"):
        panes_list = []
        for i in range(args.num_seeds):

            pane_str = template.format(transfer_type,
                                       env_name,
                                       args.process_num,
                                       i * 1000,
                                       dir_name,
                                       args.traj_num,
                                       args.iteration_num,
                                       args.variety_list)

            pane_str = "sleep {}s && ".format(sleep_inverval * sleep_index) + pane_str
            dir_name += 1
            sleep_index += 1

            if args.conda_name is not None:
                pane_str = "source activate {} && ".format(args.conda_name) + pane_str

            if first:
                pane_str = "rm -rf ./result && " + pane_str
                first = False

            panes_list.append(pane_str)

        config["windows"].append({
            "window_name": "{}-{}".format(env_name, transfer_type),
            "panes": panes_list
        })

yaml.dump(config, open("run_all.yaml", "w"), default_flow_style=False)
