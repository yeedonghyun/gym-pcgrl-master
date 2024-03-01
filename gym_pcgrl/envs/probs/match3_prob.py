from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward

class Match3Problem(Problem):
    class Pos():
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def __init__(self):
        super().__init__()
        self._width = 9
        self._height = 7
        self._prob = {"empty": 0.5, "solid":0.5}
        self._border_tile = "solid"

        self._desired_swap_potential = 500

        self.vector = [[self.Pos(-1, 0), self.Pos(-2, 0)], [self.Pos(-1, 0), self.Pos(1, 0)], [self.Pos(1, 0), self.Pos(2, 0)], 
                    [self.Pos(0, -1), self.Pos(0, -2)], [self.Pos(0, -1), self.Pos(0, 1)], [self.Pos(0, 1), self.Pos(0, 2)]]
        
        self.candidate_dir = [[self.Pos(1, 0), self.Pos(0, -1), self.Pos(0, 1)], [self.Pos(0, 1), self.Pos(0, -1)], 
                              [self.Pos(-1, 0), self.Pos(0, 1), self.Pos(0, -1)], [self.Pos(0, 1), self.Pos(1, 0), self.Pos(-1, 0)], 
                              [self.Pos(-1, 0), self.Pos(1, 0)], [self.Pos(-1, 0), self.Pos(1, 0), self.Pos(0, -1)]]
        
        self._rewards = {
            "swap_potential": 1,
        }

    def get_tile_types(self):
        return ["empty", "solid"]

    def get_stats(self, map):

        map_stats = {
            "swap_potential": 0,
        }

        for x in range(self._width):
            for y in range(self._height):
                if map[y][x] != 1 :
                    for i in range(len(self.vector)) :
                        cell_1 = self.Pos(x + self.vector[i][0].x, y + self.vector[i][0].y)
                        cell_2 = self.Pos(x + self.vector[i][1].x, y + self.vector[i][1].y)

                        if self.isvalid_cell(cell_1, map) and self.isvalid_cell(cell_2, map) :
                            for d in self.candidate_dir[i] :
                                candidate_cell = self.Pos(x + d.x, y + d.y)

                                if self.isvalid_cell(candidate_cell, map):
                                    map_stats["swap_potential"] += 1

        return map_stats
    
    def isvalid_cell(self, pos, map):
        if  pos.x < 0 or pos.x >= self._width or pos.y < 0 or pos.y >= self._height or map[pos.y][pos.x] == "solid":
            return False
        
        return True
    
    def get_reward(self, new_stats, old_stats):
        rewards = {
            "swap_potential": get_range_reward(new_stats["swap_potential"], old_stats["swap_potential"], self._desired_swap_potential - 20, self._desired_swap_potential + 20),
        }
        return rewards["swap_potential"] * self._rewards["swap_potential"]

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["swap_potential"] >= self._desired_swap_potential - 20 and new_stats["swap_potential"] <= self._desired_swap_potential + 20

    def get_debug_info(self, new_stats, old_stats):
        return {
            "swap_potential": new_stats["swap_potential"],
        }