import csv

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from myagent import SarsaLambdaAgent


def make_env(render_mode=None):
    """
    Create the tabular environment used for Task 2.
    """

    return gym.make(
        "FrozenLake-v1",
        is_slippery=False,
        render_mode=render_mode
    )


LAMBDAS = [0.0, 0.3, 0.6, 0.9, 1.0]
SEEDS = [0, 1, 2, 3, 4]

TOTAL_EPISODES = 1000
SMOOTHING_WINDOW = 100
TARGET_RETURN = 0.8
FINAL_WINDOW = 100


def smooth(values, window):
    """Return a moving average of the values."""

    if len(values) < window:
        return np.asarray(values)

    kernel = np.ones(window) / window

    return np.convolve(
        values,
        kernel,
        mode="valid"
    )


def run_lambda_sweep():
    """
    Train one fresh agent for every lambda and seed.

    Returns:
        Dictionary mapping each lambda to an array with shape:

        (number of seeds, number of episodes)
    """

    results = {}

    for lam in LAMBDAS:
        print(f"\nTraining lambda = {lam}")

        seed_returns = []

        for seed in SEEDS:
            print(f"  Seed {seed}")

            # Create a fresh environment for every experiment.
            env = make_env()

            # Set the environment seed.
            env.reset(seed=seed)

            agent = SarsaLambdaAgent(
                env=env,
                gamma=0.99,
                alpha=0.05,
                eps=0.1,
                lam=lam,
                total_epi=TOTAL_EPISODES,
                init_val=1.0,
                seed=seed
            )

            returns = agent.learn()
            seed_returns.append(returns)

            env.close()

        results[lam] = np.asarray(seed_returns)

    return results


def make_learning_curve_plot(results):
    """Create and save the learning-curve plot."""

    plt.figure(figsize=(10, 6))

    for lam in LAMBDAS:
        data = results[lam]

        # Smooth each seed separately.
        smoothed_runs = np.asarray([
            smooth(run, SMOOTHING_WINDOW)
            for run in data
        ])

        # Average the smoothed curves across seeds.
        mean_curve = smoothed_runs.mean(axis=0)

        # Standard deviation across seeds.
        std_curve = smoothed_runs.std(axis=0)

        # The first moving-average point represents episode 99.
        episodes = np.arange(
            SMOOTHING_WINDOW - 1,
            len(data[0])
        )

        plt.plot(
            episodes,
            mean_curve,
            label=f"lambda = {lam}"
        )

        plt.fill_between(
            episodes,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2
        )

    plt.xlabel("Episode")
    plt.ylabel("Mean return")

    plt.title(
        "SARSA(lambda) Learning Curves "
        f"({len(SEEDS)} seeds, "
        f"{SMOOTHING_WINDOW}-episode smoothing)"
    )

    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "lambda_learning_curves.png",
        dpi=200
    )

    plt.show()

    print("\nSaved plot to lambda_learning_curves.png")


def find_first_target_episode(values, target):
    """
    Find the first index where the curve reaches the target.

    Returns:
        None if the target is never reached.
    """

    reached = np.where(values >= target)[0]

    if len(reached) == 0:
        return None

    return int(reached[0])


def create_results_table(results):
    """
    Create and save the lambda comparison table.
    """

    rows = []

    for lam in LAMBDAS:
        data = results[lam]

        # Mean return across seeds for each episode.
        mean_curve = data.mean(axis=0)

        # Smooth the mean curve.
        smoothed_mean = smooth(
            mean_curve,
            SMOOTHING_WINDOW
        )

        first_target = find_first_target_episode(
            smoothed_mean,
            TARGET_RETURN
        )

        if first_target is not None:
            # Convert the smoothed-array index to the real episode number.
            first_target += SMOOTHING_WINDOW - 1

        # Average the final 100 episodes across all seeds.
        final_data = data[:, -FINAL_WINDOW:]

        mean_final_return = float(
            final_data.mean()
        )

        rows.append({
            "lambda": lam,
            "target_return": TARGET_RETURN,
            "first_episode_reaching_target": first_target,
            "mean_final_return": mean_final_return
        })

    with open(
        "lambda_results.csv",
        "w",
        newline=""
    ) as file:

        fieldnames = [
            "lambda",
            "target_return",
            "first_episode_reaching_target",
            "mean_final_return"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nLambda results")
    print("-" * 75)

    for row in rows:
        print(
            f"lambda={row['lambda']:<4} | "
            f"target episode="
            f"{str(row['first_episode_reaching_target']):<8} | "
            f"mean final return="
            f"{row['mean_final_return']:.3f}"
        )

    print("\nTarget return:", TARGET_RETURN)
    print("Saved table to lambda_results.csv")


def show_greedy_episode(lam=0.9, seed=0):
    """
    Train one agent and display one greedy ANSI episode.
    """

    env = make_env(render_mode="ansi")

    env.reset(seed=seed)

    agent = SarsaLambdaAgent(
        env=env,
        gamma=0.99,
        alpha=0.05,
        eps=0.1,
        lam=lam,
        total_epi=TOTAL_EPISODES,
        init_val=1.0,
        seed=seed
    )

    print(
        f"\nTraining greedy agent with lambda={lam}"
    )

    agent.learn()

    state, info = env.reset(seed=seed)

    total_return = 0.0
    terminated = False
    truncated = False

    for step in range(300):
        rendered_state = env.render()

        if rendered_state is not None:
            print(f"\nStep {step}")
            print(rendered_state)

        # Greedy action: exploration is disabled.
        action = agent.eps_greedy(
            state,
            exploration=False
        )

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_return += reward
        state = next_state

        if terminated or truncated:
            break

    print("\nGreedy episode return:", total_return)
    print("Number of steps:", step + 1)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    env.close()


def main():
    # Run all lambda values and seeds.
    results = run_lambda_sweep()

    # Create the learning-curve plot.
    make_learning_curve_plot(results)

    # Create the results table.
    create_results_table(results)

    # Show one trained greedy episode.
    show_greedy_episode(
        lam=0.9,
        seed=0
    )


if __name__ == "__main__":
    main()