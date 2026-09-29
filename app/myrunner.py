import csv
import numpy as np
import matplotlib.pyplot as plt

from myagent import SarsaLambdaAgent
import gymnasium as gym

# I tested againts a different game bc we shouldnt be importing our game anyway
def make_env(render_mode=None):
    return gym.make(
        "FrozenLake-v1",
        is_slippery=False,
        render_mode=render_mode
    )

LAMBDAS = [0.0, 0.3, 0.6, 0.9, 1.0]
SEEDS = [0, 1, 2, 3, 4]


SMOOTHING_WINDOW = 100
TARGET_RETURN = 1.0
FINAL_WINDOW = 100
TOTAL_EPISODES = 1000

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
        Dictionary mapping lambda values to arrays with shape:
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

            # Seed the environment.
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

        mean_curve = smoothed_runs.mean(axis=0)
        std_curve = smoothed_runs.std(axis=0)

        episodes = np.arange(len(mean_curve))

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
        f"SARSA(lambda) Learning Curves "
        f"({len(SEEDS)} seeds, {SMOOTHING_WINDOW}-episode smoothing)"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "lambda_learning_curves.png",
        dpi=200
    )

    plt.show()


def find_first_target_episode(values, target):
    """
    Find the first episode where the mean curve reaches target.

    Returns None if the target is never reached.
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

        # Mean return across seeds for every episode.
        mean_curve = data.mean(axis=0)

        # Use a smoothed curve to avoid counting one lucky episode.
        smoothed_mean = smooth(
            mean_curve,
            SMOOTHING_WINDOW
        )

        first_target = find_first_target_episode(
            smoothed_mean,
            TARGET_RETURN
        )

        # Average the final window across all seeds.
        final_data = data[:, -FINAL_WINDOW:]
        mean_final_return = float(final_data.mean())

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
            f"first target episode="
            f"{str(row['first_episode_reaching_target']):<8} | "
            f"mean final return="
            f"{row['mean_final_return']:.3f}"
        )

    print("\nSaved table to lambda_results.csv")


def show_greedy_episode(lam=0.9, seed=0):
    """
    Train one agent and display one greedy episode using ANSI rendering.
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

    print(f"\nTraining agent for greedy episode, lambda={lam}")
    agent.learn()

    state, info = env.reset(seed=seed)

    episode = []
    total_return = 0.0

    for step in range(300):
        rendered_state = env.render()

        if rendered_state is not None:
            print(f"\nStep {step}")
            print(rendered_state)

        action = agent.eps_greedy(
            state,
            exploration=False
        )

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        episode.append(
            (int(state), int(action), float(reward))
        )

        total_return += reward
        state = next_state

        if terminated or truncated:
            break

    print("\nGreedy episode return:", total_return)
    print("Number of steps:", len(episode))
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    env.close()


def main():
    results = run_lambda_sweep()

    make_learning_curve_plot(results)

    create_results_table(results)

    show_greedy_episode(
        lam=0.9,
        seed=0
    )


if __name__ == "__main__":
    main()