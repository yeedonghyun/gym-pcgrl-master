import numpy as np
import os

from model import FullyConvPolicyBigMap
from utils import get_exp_name, max_exp_idx, make_vec_envs
from stable_baselines import PPO2
from stable_baselines.results_plotter import load_results, ts2xy
from datetime import datetime

best_mean_reward, n_steps = -np.inf, 0
log_dir = './'

def callback(_locals, _globals):
    global n_steps, best_mean_reward
    if (n_steps + 1) % 100 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 100:
            best_mean_reward = max(mean_reward, best_mean_reward)
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

        _locals['self'].save(os.path.join(log_dir, 'model.pkl'))

    n_steps += 1
    return True

def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
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
    
    kwargs = {
        **kwargs,
        'render_rank': 0,
        'render': render,
    }
    env = make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs)
    model = PPO2(policy, env, verbose=1, tensorboard_log="./runs")
    
    if not logging:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name)
    else:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name, callback=callback)

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
logging = True
n_cpu = 20
kwargs = {
    'resume': False
}

if __name__ == '__main__':
    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)