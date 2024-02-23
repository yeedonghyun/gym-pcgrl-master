import os

from model import FullyConvPolicyBigMap
from utils import get_exp_name, max_exp_idx, make_vec_envs
from stable_baselines import PPO2
from datetime import datetime

log_dir = './'

def main(game, representation, experiment, steps, n_cpu, render, **kwargs):
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    resume = kwargs.get('resume', False)
    start_time = datetime.now().replace()
    policy = FullyConvPolicyBigMap

    n = max_exp_idx(exp_name)
    global log_dir

    if not resume:
        n = n + 1
    log_dir = 'runs/{}_{}_{}'.format(exp_name, n, 'log')
    if not resume:
        os.mkdir(log_dir)
    
    env = make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs)
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
render = False
n_cpu = 20
kwargs = {
    'resume': False,
    'render_rank': 0,
    'render': render
}

if __name__ == '__main__':
    main(game, representation, experiment, steps, n_cpu, render, **kwargs)