import copy
import numpy as np
import heapq

from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_certain_tile

class MazeProblem(Problem):
    def __init__(self):
        super().__init__()
        self._width = 13
        #player[0] == width
        self._height = 10
        #player[1] == height
        self._prob = {"empty": 0.6, "solid":0.38, "player":0.01, "goal":0.01}
        self._border_tile = "solid"

        self._desired_crossroads = 25
        self._desired_number_of_solids_around_goal = 2

        self._rewards = {
            "crossroads": 2,
            "players": 3,
            "goals": 3,
            "solids_around_goal" : 50,
            "regions": 5
        }

        self.dir = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]

    def get_tile_types(self):
        return ["empty", "solid", "player", "goal"]

    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())

        map_stats = {
            "crossroads": 0,
            "players": calc_certain_tile(map_locations, ["player"]),
            "goals": calc_certain_tile(map_locations, ["goal"]),
            "solids_around_goal": 0,
            "regions": calc_num_regions(map, map_locations, ["empty", "player", "goal"]),
        }
        if map_stats["players"] != 1 or map_stats["goals"] != 1 or map_stats["regions"] != 1:
            return map_stats
        
        map_stats["solids_around_goal"] = self.__is_goal_area_valid(map, map_locations["goal"][0])
        map_stats["crossroads"] = self.__a_star(map, map_locations["player"][0])

        return map_stats
    
    def get_reward(self, new_stats, old_stats):
        rewards = {
            "crossroads": get_range_reward(new_stats["crossroads"], old_stats["crossroads"], np.inf, np.inf),
            "players": get_range_reward(new_stats["players"], old_stats["players"], 1, 1),
            "goals": get_range_reward(new_stats["goals"], old_stats["goals"], 1, 1),
            "solids_around_goal" : get_range_reward(new_stats["solids_around_goal"], old_stats["solids_around_goal"], 1, 1),
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
        }
        return rewards["crossroads"] * self._rewards["crossroads"] +\
            rewards["players"] * self._rewards["players"] +\
            rewards["goals"] * self._rewards["goals"] +\
            rewards["solids_around_goal"] * self._rewards["solids_around_goal"] +\
            rewards["regions"] * self._rewards["regions"]
            
    def get_episode_over(self, new_stats, old_stats):
        return new_stats["crossroads"] >= 50 and new_stats["players"] == 1 and new_stats["goals"] == 1 and new_stats["solids_around_goal"] == 1 and new_stats["regions"] == 1

    def get_debug_info(self, new_stats, old_stats):
        return {
            "crossroads": new_stats["crossroads"], 
            "players": new_stats["players"],
            "goals": new_stats["goals"],
            "solids_around_goal" : new_stats["solids_around_goal"],
            "regions": new_stats["regions"]
        }

    def __is_goal_area_valid(self, map, player):
        if player[0] == 0 or player[0] == self._width - 1 or player[1] == 0 or player[1] == self._height - 1:
            return False

        cnt = 0
        for dir in self.dir :
            around_pos = [player[0] + dir[0], player[1] + dir[1]]
            if not self.__out_of_range(around_pos) and map[around_pos[1]][around_pos[0]] == "solid" :
                cnt += 1
        if cnt == self._desired_number_of_solids_around_goal:
            return True 
        
        return False
    
    def __out_of_range(self, pos):
        if pos[0] < 0 or pos[0] >= self._width or pos[1] < 0 or pos[1] >= self._height:
            return True
        
        return False
    
    def __a_star(self, map, start):
        def count_valid_directions(pos):
            np_pos = np.array(copy.deepcopy(pos))
            num_of_movable_dir = 0
            for direction in self.dir:
                if not self.__out_of_range(np_pos + direction) and map[np_pos[1] + direction[1]][np_pos[0] + direction[0]] == "empty" :
                    num_of_movable_dir += 1

            return num_of_movable_dir

        def move_pos(pos, dir):
            np_pos = np.array(copy.deepcopy(pos))
            while not self.__out_of_range(np_pos + dir) and map[np_pos[1]+ dir[1]][np_pos[0] + dir[0]] != "solid":
                np_pos += dir

            return tuple(np_pos)
            
        start = tuple(start)
        visited = set()
        visited.add(start)
        priority_queue = [(0, start)]
        heapq.heapify(priority_queue)
        num_dir = count_valid_directions(start)

        while priority_queue:
            current_cost, current_pos = heapq.heappop(priority_queue)
            num_dir = count_valid_directions(start) - 1

            for direction in self.dir:
                new_pos = move_pos(current_pos, direction)
                new_cost = current_cost + num_dir

                if new_pos[0] == current_pos[0] and new_pos[1] == current_pos[1] :
                    continue

                if map[new_pos[1]][new_pos[0]] == "goal":
                    return new_cost + 1
                
                if new_pos not in visited:
                    heapq.heappush(priority_queue, (new_cost, new_pos))
                    visited.add(new_pos)

        return False