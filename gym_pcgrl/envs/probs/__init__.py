from gym_pcgrl.envs.probs.match3_prob import Match3Problem
from gym_pcgrl.envs.probs.maze_prob import MazeProblem

# all the problems should be defined here with its corresponding class
PROBLEMS = {
    "match3": Match3Problem,
    "maze": MazeProblem

}
