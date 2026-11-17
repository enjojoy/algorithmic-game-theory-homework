#!/usr/bin/env python3

import numpy as np


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Generate a strategy based on the given cumulative regrets.

    If no action has a positive cumulative regret, the uniform strategy is returned.

    Parameters
    ----------
    regrets : np.ndarray
        The vector containing cumulative regret of each action

    Returns
    -------
    np.ndarray
        The generated strategy
    """

    positive_regrets = np.maximum(regrets, 0).astype(np.float64)
    total_regret = positive_regrets.sum()

    if total_regret > 0:
        return positive_regrets / total_regret

    return np.full(len(regrets), 1 / len(regrets), dtype=np.float64)


def regret_minimization(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Regret Minimization for a given number of iterations.

    The reference solution starts with uniform strategies, which form the first strategy
    profile, and updates the cumulative regrets of both players simultaneously.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of `num_iters` average strategy profiles produced by the algorithm
    """

    num_rows, num_cols = row_matrix.shape
    row_regrets = np.zeros(num_rows)
    col_regrets = np.zeros(num_cols)
    row_average = np.zeros(num_rows)
    col_average = np.zeros(num_cols)
    strategies = []

    for i in range(num_iters):
        # Zero initial regrets produce uniform strategies in the first round.
        row_strategy = regret_matching(row_regrets)
        col_strategy = regret_matching(col_regrets)

        # Evaluate both players against the same current strategy profile.
        row_payoffs = row_matrix @ col_strategy
        col_payoffs = row_strategy @ col_matrix

        # Accumulate each action's payoff minus the player's actual payoff.
        row_regrets += row_payoffs - row_strategy @ row_payoffs
        col_regrets += col_payoffs - col_strategy @ col_payoffs

        row_average += (row_strategy - row_average) / (i + 1)
        col_average += (col_strategy - col_average) / (i + 1)
        strategies.append((row_average.copy(), col_average.copy()))

    return strategies


def main() -> None:
    pass


if __name__ == '__main__':
    main()
