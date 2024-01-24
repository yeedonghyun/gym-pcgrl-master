"""
Run a trained agent and get generated maps
"""
import model
import numpy as np

from stable_baselines import PPO2
from utils import make_vec_envs
from datetime import datetime
from gym_pcgrl.envs.helper import save_image

def infer(game, representation, model_path, output_file_path, **kwargs):
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

    for i in range(kwargs.get('trials', 1)):
        obs = env.reset()
        while(1) :
            action, _ = agent.predict(obs)
            obs, _, dones, info = env.step(action)
            if kwargs.get('verbose', False):
                print(info[0])
            if dones:
                break
            
        test = []
        for o in obs :
            for b in o :
                old = [np.where(r==1)[0][0] for r in b]
                test.append(old)
        #print(info[0])
        path = output_file_path + str(i) + '_' + str(info[0]['swap_potential'])
        save_image(test, path, 2)
    
    print("----------------------------------------------------") 
    end_time = datetime.now().replace(microsecond=0)
    print("Start time  : ", start_time)
    print("End time  : ", end_time)
    print("Total inference time  : ", end_time - start_time)
    print("----------------------------------------------------") 

################################## MAIN ########################################
game = 'match3'
representation = 'wide'
model_path = 'models/{}/{}/model.pkl'.format(game, representation)
output_file_path = 'image/{}/'.format(game)
kwargs = {
    'trials': 5,
    'verbose': False
}

if __name__ == '__main__':
    infer(game, representation, model_path, output_file_path, **kwargs)