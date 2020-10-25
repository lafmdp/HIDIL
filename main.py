'''
  Main entrance to run HIDIL.
  @python version : 3.6.8
'''

from method.transfer import transfer
from utils.customized_mujuco import self_mujuco
from utils.envs import *
from utils.utils import *

if __name__ == "__main__":

    check_file_path()

    for transfer_type in args.transfer_type.split(";"):

        # different environment
        for env_name in args.env_list.split(";"):
            args.env = env_name

            source_env = get_source_env(env_name)
            ground_truth_transition = self_mujuco(env_name)

            transfer_method = transfer(args)

            # different env variety
            for variety_degree in args.variety_list.split(";"):
                variety_degree = float(variety_degree)

                args.variety_degree = variety_degree

                if transfer_type == "gravity":
                    target_env = get_new_gravity_env(variety_degree, env_name)
                elif transfer_type == "density":
                    target_env = get_new_density_env(variety_degree, env_name)
                elif transfer_type == "friction":
                    target_env = get_new_friction_env(variety_degree, env_name)
                else:
                    raise RuntimeError("Got error transfer type %s" % transfer_type)

                random_seed = args.random_seed
                generate_log("\n-------------Env name:{}, variety:{}, transfer_type:{}-------------".format(env_name,
                                                                                                            variety_degree,
                                                                                                            transfer_type))
                args.random_seed = random_seed
                transfer_method.args = args
                transfer_method.set_training_configuration(env_name, variety_degree, transfer_type)

                transfer_method.run(source_env,
                                    target_env,
                                    ground_truth_transition)
