from stable_baselines import PPO2
from utils import make_vec_envs
from gym_pcgrl.envs.helper import save_image

def inference(game, representation, **kwargs):
    env_name = '{}-{}-v0'.format(game, representation)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    model_path = 'models/{}/{}/model.pkl'.format(game, representation)
    model = PPO2.load(model_path)

    cnt = 0
    for _ in range(kwargs.get('trials', 1)):
        obs = env.reset()
        while True:
            action, _ = model.predict(obs)
            obs, _, dones, info = env.step(action)
            
            if dones:
                path = 'image/{}/'.format(game) + str(cnt) + '_' + str(info[0]['crossroads'])
                if info[0]['crossroads'] == 77:
                #if info[0]['swap_potential'] == 0:
                    continue
                
                save_image(info, path)
                cnt += 1
                break

################################## MAIN ########################################
game = 'maze'
representation = 'wide'
kwargs = {
    'trials': 100,
    'verbose': False,
    'render': False
}

if __name__ == '__main__':
    inference(game, representation, **kwargs)