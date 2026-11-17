#!/usr/bin/env python3

import numpy as np
from scipy.optimize import linprog


def find_nash_equilibrium(row_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find a Nash equilibrium in a zero-sum normal-form game using linear programming.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A strategy profile that forms a Nash equilibrium
    """

    num_rows, num_cols = row_matrix.shape

    row_result = linprog(
        c=np.append(np.zeros(num_rows), -1),
        A_ub=np.column_stack((-row_matrix.T, np.ones(num_cols))),
        b_ub=np.zeros(num_cols),
        A_eq=np.append(np.ones(num_rows), 0)[None, :],
        b_eq=[1.0],
        bounds=[(0, None)] * num_rows + [(None, None)],
        method='highs',
    )
    if not row_result.success:
        raise RuntimeError(f'Row player LP failed: {row_result.message}')

    col_result = linprog(
        c=np.append(np.zeros(num_cols), 1),
        A_ub=np.column_stack((row_matrix, -np.ones(num_rows))),
        b_ub=np.zeros(num_rows),
        A_eq=np.append(np.ones(num_cols), 0)[None, :],
        b_eq=[1.0],
        bounds=[(0, None)] * num_cols + [(None, None)],
        method='highs',
    )
    if not col_result.success:
        raise RuntimeError(f'Column player LP failed: {col_result.message}')

    return row_result.x[:-1], col_result.x[:-1]


def find_correlated_equilibrium(row_matrix: np.ndarray, col_matrix: np.ndarray) -> np.ndarray:
    """Find a correlated equilibrium in a normal-form game using linear programming.

    While the cost vector could be selected to optimize a particular objective, such as
    maximizing the sum of players’ utilities, the reference solution sets it to the zero
    vector to ensure reproducibility during testing.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    np.ndarray
        A distribution over joint actions that forms a correlated equilibrium
    """

    num_rows, num_cols = row_matrix.shape
    num_joint_actions = num_rows * num_cols
    constraints = []

    # When recommended a row, switching to any other row must not improve payoff.
    for recommended in range(num_rows):
        for alternative in range(num_rows):
            if recommended == alternative:
                continue
            gains = np.zeros((num_rows, num_cols))
            gains[recommended, :] = row_matrix[alternative, :] - row_matrix[recommended, :]
            constraints.append(gains.ravel())

    # Apply the same condition to each recommended column and alternative column.
    for recommended in range(num_cols):
        for alternative in range(num_cols):
            if recommended == alternative:
                continue
            gains = np.zeros((num_rows, num_cols))
            gains[:, recommended] = col_matrix[:, alternative] - col_matrix[:, recommended]
            constraints.append(gains.ravel())

    # Each variable is the probability of one joint action (row, column).
    result = linprog(
        c=np.zeros(num_joint_actions),
        A_ub=np.array(constraints) if constraints else None,
        b_ub=np.zeros(len(constraints)) if constraints else None,
        A_eq=np.ones((1, num_joint_actions)),
        b_eq=[1.0],
        bounds=(0, None),
        method='highs',
    )
    if not result.success:
        raise RuntimeError(f'Correlated equilibrium LP failed: {result.message}')

    return result.x.reshape(num_rows, num_cols)


def main() -> None:
    pass


if __name__ == '__main__':
    main()
