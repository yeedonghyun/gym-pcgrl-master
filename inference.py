from stable_baselines import PPO2
from utils import make_vec_envs
from gym_pcgrl.envs.helper import save_image

def inference(game, representation, **kwargs):
    env_name = '{}-{}-v0'.format(game, representation)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    model_path = 'models/{}/{}/model.pkl'.format(game, representation)
    model = PPO2.load(model_path)
    
    for i in range(kwargs.get('trials', 1)):
        obs = env.reset()
        for _ in range(1000) :
            action, _ = model.predict(obs)
            obs, _, dones, info = env.step(action)

            if dones:        
                path = 'image/{}/'.format(game) + str(i) + '_' + str(info[0]['swap_potential'])
                save_image(info[0]['terminal_observation'], path)
                break

################################## MAIN ########################################
game = 'match3'
representation = 'wide'
kwargs = {
    'trials': 5,
    'verbose': False,
    'render': False
}

if __name__ == '__main__':
    inference(game, representation, **kwargs)