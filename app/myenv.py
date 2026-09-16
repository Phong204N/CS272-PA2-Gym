"""Task 1: your own custom Gymnasium environment.

Design the world yourself. The requirements it has to meet are in the assignment
readme.

Delete this docstring and describe your own world instead.
"""

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class MyEnv(gym.Env):
    """TODO: one line on what this world is and what the agent is trying to do."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        # TODO: describe your world here -- the map, the pieces, the constants.
        self.CONST_DIRECTIONS = 4
        self.CONST_WORLD_X = 10
        self.CONST_WORLD_Y = self.CONST_WORLD_X

        self.world_state = [[]]
        self.reset_world()

        # TODO: set the two spaces. Both must be Discrete.

        self.observation_space = spaces.Box(
            low=0,high=5,shape=(self.CONST_WORLD_X,self.CONST_WORLD_Y),dtype=np.uint8
        )
        self.action_space = spaces.Discrete(self.CONST_DIRECTIONS, start=2)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

    def _get_obs(self):
        return self.world_state

    def reset(self, seed: int | None = None, options: dict | None = None):
        # This line seeds self.np_random. Without it, seeding does not work and
        # the reproducibility test fails.
        super().reset(seed=seed)

        # TODO: put the world back to its starting state.
        self.reset_world()
        self.random_spawn()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        # TODO: apply the action, with noise drawn from self.np_random.
        #
        # Return terminated=True when the episode genuinely ends -- goal reached,
        # agent died, game over. Leave truncated as False and let the TimeLimit
        # wrapper from register() handle running out of time. The agent treats
        # the two differently, and so should you.

        ##  TODO:: PSEUDO-CODE
        ##  for x in world_length
        ##      for y in world_height
        ##          head_exists = false
        ##          if cell[x][y] > 1, is snake
        ##              facing init
        ##              match cell[x][y]
        ##                  case 2 is up
        ##                      if y is 0, terminate, subtract point
        ##                      facing = cell[x][y-1]
        ##                  case 3 is right
        ##                      if x = world_length, terminate, subtract point
        ##                      facing = cell[x+1][y]
        ##                  case 4 is down
        ##                      if y = world_height, terminate, subtract point
        ##                      facing = cell[x][y+1]
        ##                  case 5 is left
        ##                      if x = 0, terminate, subtract point
        ##                      facing = cell[x-1][y]
        ##              
        ##              if facing > 1, is snake, is body
        ##              elif facing < 2, is good
        ##                  if facing is 1, is fruit
        ##                      add point
        ##                      todo increase
        ##                  
        ##                  
        ##      if head_exists is false, is dead
        ##          terminate, subtract point
        

        raise NotImplementedError

    def render(self):
        """Return a readable picture of the current state, as a string."""
        if self.render_mode != "ansi":
            return None
        # TODO: draw it. You need this for the sample episode in your report.
        raise NotImplementedError

    def close(self):
        pass

    def reset_world(self):
        self.world_state = [
            [0 for x in range(0, self.CONST_WORLD_X)]
            for y in range(0, self.CONST_WORLD_Y)
        ]

    def random_spawn(self):
        start_x = random.randint(1,self.CONST_WORLD_X-2)
        start_y = random.randint(1, self.CONST_WORLD_Y-2)

        ## TODO:: Revisit.  Directions start at 2.  2==UP. 3==RIGHT. 4==DOWN.  5==LEFT.
        ##          -  This is because 0==BLANK.  1==FRUIT.
        start_dir = random.randint(2, 2+self.CONST_DIRECTIONS-1)

        self.world_state[start_x][start_y] = start_dir


# TODO: name your environment. The id must start with "cs272/" and end with a
# version, and max_episode_steps must be large enough that a competent agent can
# finish but small enough that a lost one gives up.
register(
    id="cs272/MyEnv-v0",
    entry_point="myenv:MyEnv",
    max_episode_steps=300,
)
