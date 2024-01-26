import queue
import copy
import numpy as np

from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_certain_tile
    
class MazeProblem(Problem):
    def __init__(self):
        super().__init__()
        self._width = 11
        self._height = 7
        self._prob = {"empty": 0.6, "solid":0.36, "player":0.02, "goal":0.02}
        self._border_tile = "solid"

        self._target_crossroads = 25

        self._rewards = {
            "crossroads": 2,
            "players": 3,
            "goals": 3,
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
            "regions": calc_num_regions(map, map_locations, ["empty", "player", "goal"]),
        }
        if map_stats["players"] != 1 or map_stats["goals"] != 1:
            return map_stats

        #BFS
        p_x, p_y = map_locations["player"][0]
        visited = np.zeros((self._height, self._width))
        visited[p_y][p_x] = True

        # number of crossings per route and start position
        route = [np.array([0, 0]), np.array([p_x, p_y])]                                     

        Q = queue.Queue()
        Q.put(route)

        while not Q.empty():
            temp_Q = queue.Queue()

            while not Q.empty():
                cur_route = copy.deepcopy(Q.get())
                players_moved_by_dir = []

                for dir in self.dir:
                    player = self.__move_pos(cur_route[-1], dir, visited, map)

                    if player[0] != cur_route[-1][0] or player[1] != cur_route[-1][1] :
                        players_moved_by_dir.append(player)

                cur_route[0][0] += len(players_moved_by_dir)
                
                for pos in players_moved_by_dir:    
                    if map[pos[1]][pos[0]] == "goal":
                        map_stats["crossroads"] = cur_route[0][0]
                        return map_stats

                    temp_route = cur_route.copy()
                    temp_route.append(pos)
                    temp_Q.put(temp_route)
                    visited[pos[1]][pos[0]] = True

            Q = temp_Q

        return map_stats
    
    def get_reward(self, new_stats, old_stats):
        rewards = {
            "crossroads": get_range_reward(new_stats["crossroads"], old_stats["crossroads"], self._target_crossroads, self._target_crossroads),
            "players": get_range_reward(new_stats["players"], old_stats["players"], 1, 1),
            "goals": get_range_reward(new_stats["goals"], old_stats["goals"], 1, 1),
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
        }
        #calculate the total reward
        return rewards["crossroads"] * self._rewards["crossroads"] +\
            rewards["players"] * self._rewards["players"] +\
            rewards["goals"] * self._rewards["goals"] +\
            rewards["regions"] * self._rewards["regions"]

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["crossroads"] == self._target_crossroads and new_stats["regions"] == 1 and new_stats["players"] == 1 and new_stats["goals"] == 1

    def get_debug_info(self, new_stats, old_stats):
        return {
            "crossroads": new_stats["crossroads"], 
            "players": new_stats["players"],
            "goals": new_stats["goals"],
            "regions": new_stats["regions"]
        }
    
    def __move_pos(self, start, dir, visited, map):
        pos = copy.deepcopy(start)

        while not self.__out_of_range(pos + dir) and map[pos[1]+ dir[1]][pos[0] + + dir[0]] != "solid":
            pos += dir

        if visited[pos[1]][pos[0]]:
            return start

        return pos
    
    def __out_of_range(self, pos):
        if pos[0] < 0 or pos[0] >= self._width or pos[1] < 0 or pos[1] >= self._height:
            return True
        
        return False