import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import random
from gym_pcgrl.envs.helper import save_image, get_string_map

def generate_random_list(width, height, prob):
    elements = list(prob.keys())
    probabilities = list(prob.values())

    random_list = []

    for i in range(height):
        row = []
        for j in range(width):
            element = random.choices(elements, probabilities)[0]
            row.append(element)

        random_list.append(row)

    return random_list

class YourClass:
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def one_hot_encode(self, obs):
        old = obs[self.name]
        encoded_value = [0] * self.dim
        encoded_value[old] = 1
        obs[self.name] = encoded_value
        return obs

    def inverse_one_hot_encode(self, obs):
        encoded_value = obs[self.name]
        old = encoded_value.index(1)
        obs[self.name] = old
        return obs


def main(game):
    _width = 11
    _height = 7
    _prob = {"empty": 0.6, "solid":0.36, "player":0.02, "goal":0.02}    
    map = generate_random_list(_width, _height, _prob)

    random_list = []

    for i in range(_height):
        row = []
        for j in range(_width):
            element = random.
            row.append(element)

        random_list.append(row)


game = 'maze'

if __name__ == '__main__':
    main(game)