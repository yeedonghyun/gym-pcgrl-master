import copy
import numpy as np
import heapq

from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_certain_tile

#player[0] == width
#player[1] == height

class MazeProblem(Problem):
    def __init__(self):
        super().__init__()
        self._width = 10
        self._height = 10
        self._prob = {"empty": 0.6, "solid":0.38, "player":0.01, "goal":0.01}
        self._border_tile = "solid"

        self._desired_crossroads = 25

        self._rewards = {
            "crossroads": 2,
            "players": 3,
            "goals": 3,
            "valid_goal" : 10,
            "regions": 5
        }

        self.dir = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]

        self.n_action = 0

    def get_tile_types(self):
        return ["empty", "solid", "player", "goal"]

    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        self.n_action += 1

        map_stats = {
            "crossroads": 0,
            "players": calc_certain_tile(map_locations, ["player"]),
            "goals": calc_certain_tile(map_locations, ["goal"]),
            "valid_goal": False,
            "regions": calc_num_regions(map, map_locations, ["empty", "player", "goal"]),
        }
        if map_stats["players"] != 1 or map_stats["goals"] != 1 or map_stats["regions"] != 1 :
            return map_stats

        map_stats["valid_goal"] = self.__is_valid_goal(map, map_locations)
        if not map_stats["valid_goal"] :
            return map_stats

        map_stats["crossroads"] = self.__a_star(map, map_locations)

        return map_stats
    
    def get_reward(self, new_stats, old_stats):
        rewards = {
            "crossroads": get_range_reward(new_stats["crossroads"], old_stats["crossroads"], self._desired_crossroads, self._desired_crossroads),
            "players": get_range_reward(new_stats["players"], old_stats["players"], 1, 1),
            "goals": get_range_reward(new_stats["goals"], old_stats["goals"], 1, 1),
            "valid_goal" : get_range_reward(new_stats["valid_goal"], old_stats["valid_goal"], 1, 1),
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
        }
        return rewards["crossroads"] * self._rewards["crossroads"] +\
            rewards["players"] * self._rewards["players"] +\
            rewards["goals"] * self._rewards["goals"] +\
            rewards["valid_goal"] * self._rewards["valid_goal"] +\
            rewards["regions"] * self._rewards["regions"]
            
    def get_episode_over(self, new_stats, old_stats):
        #return new_stats["crossroads"] == self._desired_crossroads or (self.n_action + 1) % 1000 == 0
        return new_stats["crossroads"] == self._desired_crossroads
    
    def get_debug_info(self, new_stats, old_stats):
        return {
            "crossroads": new_stats["crossroads"], 
            "players": new_stats["players"],
            "goals": new_stats["goals"],
            "valid_goal" : new_stats["valid_goal"],
            "regions": new_stats["regions"]
        }

    def __is_valid_goal(self, map, map_locations):
        goal = map_locations["goal"][0]

        if goal[0] == 0 or goal[0] == self._width - 1 or goal[1] == 0 or goal[1] == self._height - 1:
            return True

        for dir in self.dir :
            if map[goal[1] + dir[1]][goal[0] + dir[0]] == "solid" :
                return True 
        
        return False
    
    def __a_star(self, map, map_locations):
        def out_of_range(pos):
            if pos[0] < 0 or pos[0] >= self._width or pos[1] < 0 or pos[1] >= self._height:
                return True
        
            return False

        def count_valid_directions(pos):
            np_pos = np.array(copy.deepcopy(pos))
            num_of_movable_dir = 0
            for direction in self.dir:
                new_pos = np.array(np_pos + direction)
                if not out_of_range(new_pos) and map[new_pos[1]][new_pos[0]] == "empty" :
                    num_of_movable_dir += 1

            return num_of_movable_dir

        def move_pos(pos, dir):
            np_pos = np.array(copy.deepcopy(pos))
            while not out_of_range(np_pos + dir) and map[np_pos[1]+ dir[1]][np_pos[0] + dir[0]] != "solid":
                np_pos += dir

            return tuple(np_pos)
            
        start = tuple(map_locations["player"][0])
        visited = set()
        visited.add(start)
        priority_queue = [(0, start)]
        heapq.heapify(priority_queue)

        while priority_queue:
            prev_cost, prev_pos = heapq.heappop(priority_queue)
            cur_cost = count_valid_directions(prev_pos) - 1

            for direction in self.dir:
                new_pos = move_pos(prev_pos, direction)
                new_cost = prev_cost + cur_cost

                if new_pos[0] == prev_pos[0] and new_pos[1] == prev_pos[1] :
                    continue

                if map[new_pos[1]][new_pos[0]] == "goal":
                    return new_cost + 1
                
                if new_pos not in visited:
                    heapq.heappush(priority_queue, (new_cost, new_pos))
                    visited.add(new_pos)

        return False