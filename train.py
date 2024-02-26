import os

from model import FullyConvPolicyBigMap
from utils import get_exp_name, max_exp_idx, make_vec_envs
from stable_baselines import PPO2
from datetime import datetime

def main(game, representation, experiment, steps, n_cpu, **kwargs):
    start_time = datetime.now().replace()

    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    n = max_exp_idx(exp_name) + 1
    log_dir = 'runs/{}_{}_{}'.format(exp_name, n, 'log')
    os.mkdir(log_dir)
    
    env_name = '{}-{}-v0'.format(game, representation)
    env = make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs)

    policy = FullyConvPolicyBigMap
    model = PPO2(policy, env, verbose=1, tensorboard_log="./runs")
    model.learn(total_timesteps=int(steps), tb_log_name=exp_name)
    model.save(os.path.join(log_dir, 'model.pkl'))

    end_time = datetime.now().replace()
    print("Start time  : ", start_time)
    print("End time  : ", end_time)
    print("Total training time  : ", end_time - start_time)

################################## MAIN ########################################
game = 'maze'
representation = 'wide'
experiment = None
steps = 50000000
n_cpu = 20
kwargs = {
    'render': False,
    'render_rank': 0
}

if __name__ == '__main__':
    main(game, representation, experiment, steps, n_cpu, **kwargs)