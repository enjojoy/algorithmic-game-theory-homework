#!/usr/bin/env python3

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


def plot_best_response_value_function(row_matrix: np.ndarray, step_size: float) -> None:
    """Plot the best response value function for the row player in a 2xN zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix, with shape (2, N)
    step_size : float
        The step size for the probability of the first action of the row player,
        between 0 (exclusive) and 1 (inclusive)
    """

    if row_matrix.ndim != 2 or row_matrix.shape[0] != 2 or row_matrix.shape[1] == 0:
        raise ValueError('row_matrix must have exactly two rows and at least one column')
    if not 0 < step_size <= 1:
        raise ValueError('step_size must be greater than 0 and at most 1')

    # Include p = 1 even when the step size does not divide 1 evenly.
    probabilities = np.append(np.arange(0, 1, step_size), 1.0)
    values = []

    for p in probabilities:
        row_strategy = np.array([p, 1 - p])
        payoffs = row_strategy @ row_matrix
        # The column player's best response minimizes the row player's payoff.
        values.append(np.min(payoffs))

    fig, ax = plt.subplots()
    ax.plot(probabilities, values)
    ax.set_xlabel("Probability p of the row player's first action")
    ax.set_ylabel("Row player's payoff against a best response")
    ax.set_title('Best response value function')
    ax.set_xlim(0, 1)
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def verify_support(
    matrix: np.ndarray, row_support: np.ndarray, col_support: np.ndarray
) -> np.ndarray | None:
    """Construct a system of linear equations to check whether there
    exists a candidate for a Nash equilibrium for the given supports.

    The returned strategy of the opponent (the column player of `matrix`) makes
    all actions in `row_support` equally good. This function is not supposed to
    check if actions outside of the given support lead to a higher payoff.

    To compute the strategy of the row player, call this function with the transposed
    payoff matrix of the column player and with the two supports swapped.

    The reference implementation uses `scipy.optimize.linprog`
    with the default solver -- 'highs'. You can find more information at
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of the player whose actions correspond to rows
    row_support : np.ndarray
        The support of the player whose actions correspond to rows of `matrix`
    col_support : np.ndarray
        The support of the opponent whose actions correspond to columns of `matrix`

    Returns
    -------
    np.ndarray | None
        Probabilities of the opponent's actions in `col_support`, if they exist, otherwise `None`
    """

    num_probabilities = len(col_support)

    # The unknowns are the opponent's probabilities, followed by a common payoff v.
    # For each supported row i, require sum_j matrix[i, j] * q[j] - v = 0.
    coefficients = np.zeros((len(row_support) + 1, num_probabilities + 1))
    coefficients[:-1, :-1] = matrix[np.ix_(row_support, col_support)]
    coefficients[:-1, -1] = -1

    # The opponent's probabilities must sum to 1.
    coefficients[-1, :-1] = 1
    constants = np.zeros(len(row_support) + 1)
    constants[-1] = 1

    # Probabilities are nonnegative; the common payoff can also be negative.
    bounds = [(0, None)] * num_probabilities + [(None, None)]
    result = linprog(
        c=np.zeros(num_probabilities + 1),
        A_eq=coefficients,
        b_eq=constants,
        bounds=bounds,
        method='highs',
    )

    return result.x[:-1] if result.success else None


def support_enumeration(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Support Enumeration to find Nash equilibria in a two-player game.

    As in the lecture slides, this finds all equilibria for nondegenerate games.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of strategy profiles corresponding to found Nash equilibria
    """

    num_rows, num_cols = row_matrix.shape
    row_supports = [
        np.array(support)
        for size in range(1, num_rows + 1)
        for support in combinations(range(num_rows), size)
    ]
    col_supports = [
        np.array(support)
        for size in range(1, num_cols + 1)
        for support in combinations(range(num_cols), size)
    ]
    equilibria = []

    for row_support in row_supports:
        for col_support in col_supports:
            col_probs = verify_support(row_matrix, row_support, col_support)
            if col_probs is None:
                continue

            row_probs = verify_support(col_matrix.T, col_support, row_support)
            if row_probs is None:
                continue

            # Actions outside the selected supports have probability zero.
            row_strategy = np.zeros(num_rows, dtype=np.float64)
            col_strategy = np.zeros(num_cols, dtype=np.float64)
            row_strategy[row_support] = row_probs
            col_strategy[col_support] = col_probs

            # Reject candidates where either player can gain by switching actions.
            row_payoffs = row_matrix @ col_strategy
            col_payoffs = row_strategy @ col_matrix
            if np.any(row_payoffs > row_payoffs[row_support[0]] + 1e-8):
                continue
            if np.any(col_payoffs > col_payoffs[col_support[0]] + 1e-8):
                continue

            # Different support pairs can produce the same equilibrium.
            if not any(
                np.allclose(row_strategy, row, rtol=0, atol=1e-8)
                and np.allclose(col_strategy, col, rtol=0, atol=1e-8)
                for row, col in equilibria
            ):
                equilibria.append((row_strategy, col_strategy))

    return equilibria


def main() -> None:
    # Matching pennies: a 2x2 zero-sum game from the test data.
    row_matrix = np.array([[1, -1], [-1, 1]], dtype=np.float64)
    plot_best_response_value_function(row_matrix, step_size=0.01)


if __name__ == '__main__':
    main()
