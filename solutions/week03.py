#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def compute_deltas(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute players' incentives to deviate from their strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
  The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        Each player's incentive to deviate
    """

    #row payoff against col strategy

    row_payoff = row_matrix @ col_strategy
    best_row_payoff = max(row_payoff)
    current_row_payoff = row_strategy @ row_payoff

    row_delta= best_row_payoff-current_row_payoff

    col_payoff = row_strategy@col_matrix
    best_col_payoff = max(col_payoff)
    current_col_payoff=col_strategy@col_payoff

    col_delta=best_col_payoff-current_col_payoff

    return np.array([row_delta, col_delta])


def compute_nash_conv(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.float64:
    """Compute the NashConv value of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The NashConv value of the given strategy profile
    """
    deltas=compute_deltas(row_matrix, col_matrix, row_strategy, col_strategy)

    return deltas.sum(dtype=np.float64)



def compute_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.float64:
    """Compute the exploitability of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The exploitability value of the given strategy profile
    """

    nash_conv=compute_nash_conv(row_matrix, col_matrix, row_strategy, col_strategy)

    return nash_conv/2


def fictitious_play(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int, naive: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Fictitious Play for a given number of iterations.

    Although any averaging method is valid, the reference solution updates the
    average strategy vectors using a moving average. Therefore, it is recommended
    to use the same averaging method to avoid numerical discrepancies during testing.

    The reference solution initializes both players' beliefs to uniform strategies
    and uses the best responses to these beliefs as the first strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for
    naive : bool
        Whether to calculate the best response against the last
        opponent's strategy or the average opponent's strategy

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of `num_iters` average strategy profiles produced by the algorithm
    """

    num_rows, num_cols = row_matrix.shape
    row_belief = np.full(num_rows, 1 / num_rows)
    col_belief = np.full(num_cols, 1 / num_cols)
    row_average = np.zeros(num_rows)
    col_average = np.zeros(num_cols)
    strategies = []

    for i in range(num_iters):
        # Both players respond to the beliefs from the previous round.
        row_payoff = row_matrix @ col_belief
        col_payoff = row_belief @ col_matrix

        row_strategy = np.zeros(num_rows)
        col_strategy = np.zeros(num_cols)
        row_strategy[np.argmax(row_payoff)] = 1.0
        col_strategy[np.argmax(col_payoff)] = 1.0

        # Average only the strategies actually played, excluding initial beliefs.
        row_average += (row_strategy - row_average) / (i + 1)
        col_average += (col_strategy - col_average) / (i + 1)
        strategies.append((row_average.copy(), col_average.copy()))

        if naive:
            row_belief = row_strategy
            col_belief = col_strategy
        else:
            row_belief = row_average
            col_belief = col_average

    return strategies


def plot_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    strategies: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> list[np.float64]:
    """Compute and plot the exploitability of a sequence of strategy profiles.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    strategies : list[tuple[np.ndarray, np.ndarray]]
        The sequence of strategy profiles
    label : str
        The name of the algorithm that produced `strategies`

    Returns
    -------
    list[np.float64]
        A sequence of exploitability values, one for each strategy profile
    """

    values = []

    for row_strategy, col_strategy in strategies:
        value = compute_exploitability(
            row_matrix, col_matrix, row_strategy, col_strategy
        )
        values.append(value)

    plt.plot(range(1, len(values) + 1), values, label=label)
    plt.xlabel('Iteration')
    plt.ylabel('Exploitability')
    plt.legend()
    plt.grid(True)

    return values


def main() -> None:
    # Compare both versions on the matching-pennies game.
    row_matrix = np.array([[1, -1], [-1, 1]], dtype=np.float64)
    col_matrix = -row_matrix

    plt.figure()
    for naive in (False, True):
        strategies = fictitious_play(row_matrix, col_matrix, num_iters=1000, naive=naive)
        label = 'Naive fictitious play' if naive else 'Fictitious play'
        plot_exploitability(row_matrix, col_matrix, strategies, label)

    plt.title('Matching pennies: exploitability over time')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
