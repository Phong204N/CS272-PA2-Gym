"""Task 2: SARSA(lambda) with eligibility traces.

Do not change the class name or the constructor signature -- the grading harness
constructs this class directly, and it will hand you an environment you have
never seen. Read the sizes off the spaces and never assume anything about what a
state number means. Do not import myenv from this file.
"""

from typing import Any

import numpy as np
import gymnasium as gym
import random

ACCUMULATING = "accumulating"
REPLACING = "replacing"


def argmax_action(values: np.ndarray, rng: np.random.Generator) -> int:
    """Return the index of the largest value, breaking ties uniformly at random.

    Ties are not an edge case here. The table starts uniform, so on the first
    visit to a state every action is tied, and a plain np.argmax would commit
    every state in the table to action 0.

    Args:
        values: the q-values of one state, shape (n_actions,)
        rng: the agent's random generator

    Returns:
        int: an action
    """
    max_value = np.max(values) #find the max value
    best_actions = np.flatnonzero(values == max_value) #find best actions tied to max value
    return int(rng.choice(best_actions)) #choose randomly

class SarsaLambdaAgent:
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99, #discount factor, future rewards value more
        alpha: float = 0.05, #learning rate
        eps: float = 0.1, #eps value
        lam: float = 0.9, #eligibility-trace parameter λ
        trace: str = ACCUMULATING,
        total_epi: int = 5_000, 
        init_val: float = 1.0, #initial value for Q table
        seed: int | None = None, #randomness control
    ) -> None:
        """
        Args:
            env: any tabular gym environment. Both spaces are Discrete.
            gamma: discount factor.
            alpha: learning rate.
            eps: exploration rate for a plain (non-decaying) epsilon-greedy.
            lam: the lambda of SARSA(lambda), in [0, 1]. At 0 this must reduce
                to ordinary one-step SARSA.
            trace: "replacing" or "accumulating".
            total_epi: number of training episodes.
            init_val: value every q(s,a) starts at. Setting this at or slightly
                above the best achievable return makes every untried action look
                good, which drives systematic exploration -- on a sparse-reward
                environment that is often what makes learning possible at all.
            seed: seed for the agent's own randomness, for reproducible runs.
        """
        if trace not in (ACCUMULATING, REPLACING):
            raise ValueError(f"unknown trace type: {trace}")

        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.lam = lam
        self.trace = trace
        self.total_epi = total_epi
        self.init_val = init_val
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.q = self.init_qtable(init_val)

    def init_qtable(self, init_val: float = 0.0) -> np.ndarray:
        """Build the q table, shape (n_states, n_actions), filled with init_val."""
        return np.full((self.n_states, self.n_actions), init_val, dtype=float)
        

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        # two paths, explore and exploitation
        #exploration: choose any action randomly
        #exploitation: choose the best Q value using argmax_action

        # Exploration:
        if exploration and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))

        #Exploitation:
        state_values = self.q[state]

        return argmax_action(state_values, self.rng)

    def learn(self) -> list[float]:
        """Run SARSA(lambda) for self.total_epi episodes, updating self.q.

        Returns:
            list[float]: the undiscounted return of each training episode, in
            order. myrunner.py plots these.
        returns = []

        for each episode:
            reset environment
            create a zero eligibility-trace table
            choose initial action with eps_greedy()

            while episode is not over:
                take action
                receive next_state, reward, terminated, truncated

                if terminal:
                    calculate terminal TD error
                else:
                    choose next_action with eps_greedy()
                    calculate SARSA TD error

                update the current eligibility trace
                update every Q-value using the traces
                decay the traces

                move to next state and action
                add reward to episode return

            append episode return to returns

            return returns
        """
        returns = []

        for _ in range(self.total_epi):
            state, info = self.env.reset()

            trace_table = np.zeros(
                (self.n_states, self.n_actions),
                dtype=float
            )

            action = self.eps_greedy(state, exploration=True)

            terminated = False
            truncated = False
            episode_return = 0.0

            while not terminated and not truncated:
                next_state, reward, terminated, truncated, info = (
                    self.env.step(action)
                )

                episode_return += reward
                current_state_val = self.q[state, action]

                if terminated:
                    td_error = reward - current_state_val

                else:
                    next_action = self.eps_greedy(
                        next_state,
                        exploration=True
                    )

                    next_state_val = self.q[next_state, next_action]

                    td_error = (
                        reward
                        + self.gamma * next_state_val
                        - current_state_val
                    )

                if self.trace == ACCUMULATING:
                    trace_table[state, action] += 1
                else:
                    trace_table[state, action] = 1

                self.q += self.alpha * td_error * trace_table

                trace_table *= self.gamma * self.lam

                if not terminated and not truncated:
                    state = next_state
                    action = next_action

            returns.append(float(episode_return))

        return returns
        

    def best_run(self, max_steps: int = 300) -> tuple[list[tuple[int, int, float]], bool]:
        state, info = self.env.reset()
        episode = []

        for _ in range(max_steps):
            action = self.eps_greedy(state,exploration=False)

            next_state, reward, terminated, truncated, info = (self.env.step(action))

            episode.append((int(state), int(action), float(reward)))

            state = next_state

            if terminated or truncated:
                return episode, terminated

        return episode, False


class RandomAgent(SarsaLambdaAgent):
    """The baseline your agent has to beat. Already written; do not change it."""

    def learn(self) -> list[float]:
        returns = []
        for _ in range(self.total_epi):
            self.env.reset()
            total = 0.0
            while True:
                action = int(self.rng.integers(self.n_actions))
                _, reward, terminated, truncated, _ = self.env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
        return returns
