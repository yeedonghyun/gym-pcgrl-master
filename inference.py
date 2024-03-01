import numpy as np

from stable_baselines import PPO2
from utils import make_vec_envs
from datetime import datetime
from gym_pcgrl.envs.helper import save_image

def inference(game, representation, model_path, output_file_path, **kwargs):
    start_time = datetime.now().replace()

    env_name = '{}-{}-v0'.format(game, representation)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)

    model = PPO2.load(model_path)
    
    for i in range(kwargs.get('trials', 1)):
        obs = env.reset()
        for j in range(1000) :
            action, _ = model.predict(obs)
            obs, _, dones, info = env.step(action)

            if dones:        
                path = output_file_path + str(i) + '_' + str(info[0]['crossroads'])
                save_image(info[0]['terminal_observation'], path)
                break

    end_time = datetime.now().replace()
    print("Start time  : ", start_time)
    print("End time  : ", end_time)
    print("Total inference time  : ", end_time - start_time)

################################## MAIN ########################################
game = 'maze'
representation = 'wide'
model_path = 'models/{}/{}/model.pkl'.format(game, representation)
output_file_path = 'image/{}/'.format(game)
kwargs = {
    'trials': 5,
    'verbose': False,
    'render': False
}

if __name__ == '__main__':
    inference(game, representation, model_path, output_file_path, **kwargs)