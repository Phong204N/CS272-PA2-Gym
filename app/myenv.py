"""
This environment emulates the classic game of snake.
The "snake" is a series of tiles with a head and a body. 
It can move in cardinal directions and grow.

To play, an agent must select 1 of 4 actions for every step.  Up, Down, Left, or Right.
This selects where the head moves next.  The rest of the body follows sequentially.  

Fruits are items that spawn and add an extra length of 1 tile to the snake when consumed.
At every step, if no previous fruits are present, a capped, random number of fruits will spawn.

The goal is for the agent to reach a maximum length, filling up all the squares, without a collision.
A collision with a wall/boundary, or its own body will terminate the game.

In a normal snake game, a human player would also be challenged by the ability of their reaction to avoid collision while planning their route. 
Since the agent needs to be "plugged" into the game, observing the world and being able to act at each step, this is not an issue.
"""

from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class MyEnv(gym.Env):
    """This environment resembles the classic snake game where an agent must choose 1 of 4 directions and collect 'fruits' to grow while avoiding collisions."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        ##  The map is a square grid of 10x10 tiles.  
        ##      -   The observation space is a 10x10 box.
        ##  The player is a series of connected tiles, or "the snake".
        ##  The snake traverses 1 tile per step, in any 4 cardinal directions, except opposite of it's current direction.
        ##     -    The action space is a discrete space of 4.
        ##  "Fruits" take up a tile each, and spawn via a random function.
        self.CONST_DIRECTIONS = 4
        self.CONST_WORLD_X = 10
        self.CONST_WORLD_Y = self.CONST_WORLD_X

        self.CONST_DIR_MAP = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        ##  Initialization
        self.prev_dir = -1
        self.world_state = [[]]
        self.snake = deque([]) #coordinates of the snake
        self.reset_world()

        # each board cell stores a value
        ##  0: BLANK
        ##  1: FRUIT
        ##  2: HEAD
        ##  3: BODY
        ##  4: TAIL
        self.observation_space = spaces.Box(
            low=0,high=4,shape=(self.CONST_WORLD_X,self.CONST_WORLD_Y),dtype=np.uint8
        )
        self.action_space = spaces.Discrete(self.CONST_DIRECTIONS)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

    def _get_obs(self):
        return self.world_state #returns the current board
    def _get_info(self):
        return {"snake_deque": self.snake, "prev_dir": self.prev_dir}

    def reset(self, seed: int | None = None, options: dict | None = None):
        # This line seeds self.np_random. Without it, seeding does not work and
        # the reproducibility test fails.
        super().reset(seed=seed)

        self.reset_world()
        self.random_spawn()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        reward = 0
        terminated = False
        truncated = False

        ## Tracking world status.
        status_count = self.world_status() #count how many cells contain blank, fruit, head, etc
        if status_count[0] == 0: #no blank squares left, win statement
            terminated = True
            reward = 1
        ## Spawning new fruit if none exists.
        if status_count[1] == 0:
            ## Spawn random amount of fruit from at least 1 to half of the current length.  If the number of blanks left are less, fill up those blanks instead.
            fruit_count = min(status_count[0], self.np_random.integers(1, int((status_count[2] + status_count[3] + status_count[4]) / 2) + 1))
            self.spawn_random_fruit(fruit_count)

        ##  Moving the snake.
        ##  Preventing the snake from turning into itself.
        ##  For example, if it was moving down, it can not suddenly move up.
        if action == 0: ## UP
            if self.prev_dir == 1:
                action = 1
        elif action == 1: ## DOWN
            if self.prev_dir == 0:
                action = 0
        elif action == 2: ## LEFT
            if self.prev_dir == 3:
                action = 3
        elif action == 3: ## RIGHT
            if self.prev_dir == 2:
                action = 2
        self.prev_dir = action

        ##  Set new head position based on the action taken.
        new_pos = self.snake[-1]
        if action == 0: ## UP
            new_pos = (self.snake[-1][0], self.snake[-1][1]-1)
        elif action == 1: ## DOWN
            new_pos = (self.snake[-1][0], self.snake[-1][1]+1)
        elif action == 2: ## LEFT
            new_pos = (self.snake[-1][0]-1, self.snake[-1][1])
        elif action == 3: ## RIGHT
            new_pos = (self.snake[-1][0]+1, self.snake[-1][1])

        ##  Update the world state based on the new head position.
        ##  Case: Boundary=DEAD
        if new_pos[0] < 0 or new_pos[0] >= self.CONST_WORLD_X or new_pos[1] < 0 or new_pos[1] >= self.CONST_WORLD_Y:
            
            #I commented this part bc i think unnecessary since snake is dead you dont care about dropping the tail or creating the tail? game reset will take care of it
            
            # # Remove the old tail
            # old_tail = self.snake.popleft()
            # self.world_state[old_tail[0]][old_tail[1]] = 0

            
            # # The old head becomes body
            # old_head = self.snake[-1]
            # self.world_state[old_head[0]][old_head[1]] = 3

            # # Mark the new tail
            # new_tail = self.snake[0]
            # self.world_state[new_tail[0]][new_tail[1]] = 4

            terminated = True
            reward = -1
        ##  Case: Blank=MOVE
        elif self.world_state[new_pos[0]][new_pos[1]] == 0:
            # Remove the old tail
            old_tail = self.snake.popleft()
            self.world_state[old_tail[0]][old_tail[1]] = 0

            # The old head becomes body
            old_head = self.snake[-1]
            self.world_state[old_head[0]][old_head[1]] = 3

            # Add the new head
            self.snake.append(new_pos)
            self.world_state[new_pos[0]][new_pos[1]] = 2

            # Mark the new tail
            new_tail = self.snake[0]
            self.world_state[new_tail[0]][new_tail[1]] = 4
        ##  Case: Fruit=EAT
        elif self.world_state[new_pos[0]][new_pos[1]] == 1:
            # Old head becomes body
            old_head = self.snake[-1]
            self.world_state[old_head[0]][old_head[1]] = 3

            # Add the new head
            self.snake.append(new_pos)
            self.world_state[new_pos[0]][new_pos[1]] = 2
            
            reward = 1
        ##  Case: Body=DEAD
        elif self.world_state[new_pos[0]][new_pos[1]] == 3:
            # Remove the old tail
            # again unnecessary
            
            # old_tail = self.snake.popleft()
            # self.world_state[old_tail[0]][old_tail[1]] = 0

            # # The old head becomes body
            # old_head = self.snake[-1]
            # self.world_state[old_head[0]][old_head[1]] = 3

            # # Add the new head
            # self.snake.append(new_pos)
            # self.world_state[new_pos[0]][new_pos[1]] = 2

            # # Mark the new tail
            # new_tail = self.snake[0]
            # self.world_state[new_tail[0]][new_tail[1]] = 4

            terminated = True
            reward = -1

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """Return a readable picture of the current state, as a string."""
        if self.render_mode != "ansi":
            return None

        TL_CORNER = "╔"
        TR_CORNER = "╗"
        BL_CORNER = "╚"
        BR_CORNER = "╝"
        HORIZONTAL = "═"
        VERTICAL = "║"
        BLANK = "."
        FRUIT = "@"
        HEAD = "H"
        BODY = "B"
        TAIL = "T"

        rendered = TL_CORNER + HORIZONTAL * self.CONST_WORLD_X + TR_CORNER + "\n"
        for y in range(0, self.CONST_WORLD_Y):
            rendered += VERTICAL
            for x in range(0, self.CONST_WORLD_X):
                if self.world_state[x][y] == 0:
                    rendered += BLANK
                elif self.world_state[x][y] == 1:
                    rendered += FRUIT
                elif self.world_state[x][y] == 2:
                    rendered += HEAD
                elif self.world_state[x][y] == 3:
                    rendered += BODY
                elif self.world_state[x][y] == 4:
                    rendered += TAIL
            rendered += VERTICAL + "\n"
        rendered += BL_CORNER + HORIZONTAL * self.CONST_WORLD_X + BR_CORNER + "\n"
        rendered += VERTICAL + f"Last Move: {self.CONST_DIR_MAP.get(self.prev_dir, 'UNKNOWN')}" + "\n"
        rendered += BL_CORNER + HORIZONTAL * self.CONST_WORLD_X + BR_CORNER

        return rendered

    def close(self):
        pass

    def world_status(self) -> tuple[list[int], tuple[int,int], tuple[int,int]]:
        status_count = [0,0,0,0,0]
        # head_pos = (-1,-1)
        # tail_pos = (-1,-1)
        for y in range(0, self.CONST_WORLD_Y):
            for x in range(0, self.CONST_WORLD_X):
                status_count[self.world_state[x][y]] += 1
                # if self.world_state[x][y] == 2:
                #     head_pos = (x, y)
                # elif self.world_state[x][y] == 4:
                #     tail_pos = (x, y)
        # return status_count, head_pos, tail_pos
        return status_count

    def reset_world(self):
        self.prev_dir = -1
        self.world_state = [
            [0 for x in range(0, self.CONST_WORLD_X)]
            for y in range(0, self.CONST_WORLD_Y)
        ]
        self.snake = deque([])

    def random_spawn(self):
        start_x = self.np_random.integers(1, self.CONST_WORLD_X-2)
        start_y = self.np_random.integers(1, self.CONST_WORLD_Y-2)

        self.world_state[start_x][start_y] = 2
        self.world_state[start_x-1][start_y] = 4
        self.snake.append((start_x, start_y))
        self.snake.appendleft((start_x-1, start_y))

    def spawn_random_fruit(self, fruit_count:int): 
        for i in range(0, fruit_count):
            fruit_x = self.np_random.integers(1, self.CONST_WORLD_X-1)
            fruit_y = self.np_random.integers(1, self.CONST_WORLD_Y-1)

            if self.world_state[fruit_x][fruit_y] == 0: #place at the blank cell
                self.world_state[fruit_x][fruit_y] = 1 #occupied
            else:
                i -= 1


# TODO: name your environment. The id must start with "cs272/" and end with a
# version, and max_episode_steps must be large enough that a competent agent can
# finish but small enough that a lost one gives up.
register(
    id="cs272/PA2-Snake-v0",
    entry_point="myenv:MyEnv",
    max_episode_steps=300,
)