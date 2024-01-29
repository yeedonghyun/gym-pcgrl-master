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

def is_all_cells_have_spawn_routes(map):
    height = len(map)
    width = len(map[0])
    visited = [[False] * width for _ in range(height)]

    def connected_cells(y, x, moved_x):
        if y < 0 or y >= height or x < 0 or x >= width:
            return 

        if visited[y][x] or map[y][x] == 1:
            return 

        visited[y][x] = True

        connected_cells(y-1, x)
        connected_cells(y+1, x)

        if not moved_x:
            connected_cells(y, x-1)
            connected_cells(y, x+1)

    for x in range(width):
        connected_cells(0, x, False)

    for y in range(height):
        for x in range(width):
            if not visited[y][x] and map[y][x] == 0:
                return False

    return True


def main(game):
    _width = 11
    _height = 7
    _prob = {"empty": 0.6, "solid":0.4}    
    map = generate_random_list(_width, _height, _prob)

    example_board = [
        [0, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 1, 1]
    ]

    result = is_all_cells_have_spawn_routes(example_board)
    print(result)

game = 'maze'

if __name__ == '__main__':
    main(game)