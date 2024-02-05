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

        connected_cells(y+1, x, False)
        if not moved_x:
            connected_cells(y, x-1, True)
            connected_cells(y, x+1, True)

    for x in range(width):
        connected_cells(0, x, True)

    for y in range(height):
        for x in range(width):
            if not visited[y][x] and map[y][x] == 0:
                return False

    return True

import heapq

def a_star_algorithm(grid, start):
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    start = tuple(start)
    goal = next((i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if cell == "goal")

    visited = set()
    priority_queue = [(0, start)]
    heapq.heapify(priority_queue)
    parent = {start: None}

    while priority_queue:
        current_cost, current_pos = heapq.heappop(priority_queue)

        if current_pos == goal:
            path = []
            while current_pos:
                path.append(current_pos)
                current_pos = parent[current_pos]
            return path[::-1]

        if current_pos in visited:
            continue

        visited.add(current_pos)

        for direction in directions:
            new_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])

            if 0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols and grid[new_pos[0]][new_pos[1]] != "solid":
                new_cost = current_cost + 1 + heuristic(new_pos)
                if new_pos not in visited or new_cost < heapq.heappop(priority_queue)[0]:
                    heapq.heappush(priority_queue, (new_cost, new_pos))
                    parent[new_pos] = current_pos

    return None  # No path found

# Example Usage:
map_example = [
    ["empty", "empty", "empty", "solid", "empty"],
    ["solid", "solid", "empty", "solid", "empty"],
    ["player", "empty", "solid", "empty", "goal"],
    ["empty", "solid", "empty", "solid", "empty"],
    ["empty", "empty", "empty", "empty", "empty"]
]

start_position = (2, 0)
path = a_star_algorithm(map_example, start_position)

if path:
    print("Shortest Path Found:", path)
else:
    print("No path found.")

def main(game):
    _width = 11
    _height = 7
    _prob = {"empty": 0.6, "solid":0.4}    
    map = generate_random_list(_width, _height, _prob)

    example_board = [
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1]
    ]

    result = is_all_cells_have_spawn_routes(example_board)
    print(result)

game = 'maze'

if __name__ == '__main__':
    main(game)