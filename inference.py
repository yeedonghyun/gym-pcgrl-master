"""
Run a trained agent and get generated maps
"""
import model
import time

from stable_baselines import PPO2
from utils import make_vec_envs
from datetime import datetime

def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    start_time = datetime.now().replace(microsecond=0)
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 10
    kwargs['render'] = False

    agent = PPO2.load(model_path)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    obs = env.reset()

    success_cnt = 0
    for i in range(kwargs.get('trials', 1)):
        for _ in range(500):
            action, _ = agent.predict(obs)
            obs, _, dones, info = env.step(action)
            if kwargs.get('verbose', False):
                print(info[0])
            if dones:
                success_cnt += 1
                break
    
    print(success_cnt)
    end_time = datetime.now().replace(microsecond=0)
    print("Total training time  : ", end_time - start_time)

################################## MAIN ########################################
game = 'match3'
representation = 'wide'
model_path = 'models/{}/{}/model_1.pkl'.format(game, representation)
kwargs = {
    'change_percentage': 0.4,
    'trials': 1000,
    'verbose': False
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)