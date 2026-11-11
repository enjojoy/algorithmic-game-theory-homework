#!/usr/bin/env python3

import numpy as np


def evaluate_general_sum(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute the expected utility of each player in a general-sum game.

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
        A vector of expected utilities of the players
    """
    strategy_weight=np.outer(row_strategy, col_strategy)
    u_col=np.sum(col_matrix*strategy_weight)
    u_row=np.sum(row_matrix*strategy_weight)

    return np.array([u_row,u_col])




def evaluate_zero_sum(
    row_matrix: np.ndarray, 
    row_strategy: np.ndarray, 
    col_strategy: np.ndarray
) -> np.ndarray:
    """Compute the expected utility of each player in a zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """

    strategy_weight=np.outer(row_strategy, col_strategy)
    u_row=np.sum(row_matrix*strategy_weight)
    return np.array([u_row, -u_row])



def calculate_best_response_against_row(
    col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the column player against the row player.

    Parameters
    ----------
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.ndarray
        The column player's best response
    """

    col_payoffs=row_strategy@col_matrix

    purebr=np.argmax(col_payoffs)

    best = np.zeros(len(col_payoffs), dtype=np.float64)

    best[purebr]=1.0

    return best



def calculate_best_response_against_col(
    row_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the row player against the column player.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        The row player's best response
    """

    row_payoffs=row_matrix@col_strategy

    purebr=np.argmax(row_payoffs)

    best=np.zeros(len(row_payoffs), dtype=np.float64)

    best[purebr]=1.0

    return best





def evaluate_row_against_best_response(
    row_matrix: np.ndarray, 
    col_matrix: np.ndarray, 
    row_strategy: np.ndarray
) -> np.float64:
    """Compute the utility of the row player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.float64
        The expected utility of the row player
    """

    col_best_res=calculate_best_response_against_row(col_matrix, row_strategy)


    return evaluate_general_sum(row_matrix, col_matrix, row_strategy, col_best_res)[0]





    


def evaluate_col_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.float64:
    """Compute the utility of the column player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The expected utility of the column player
    """
    row_best_res=calculate_best_response_against_col(row_matrix, col_strategy)


    return evaluate_general_sum(row_matrix, col_matrix, row_best_res, col_strategy)[1]







def find_strictly_dominated_actions(matrix: np.ndarray) -> np.ndarray:
    """Find strictly dominated actions for the given normal-form game.

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players

    Returns
    -------
    np.ndarray
        Indices of strictly dominated actions
    """

    dominated = []

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if j == i:
                continue
            if np.all(matrix[j]>matrix[i]):
                dominated.append(i)
                break

    return np.array(dominated, dtype=int)


def iterated_removal_of_dominated_strategies(
    row_matrix: np.ndarray, 
    col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Iterated Removal of Dominated Strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Four-tuple of reduced row and column payoff matrices, and remaining row and column actions
    """
    row_actions=np.arange(row_matrix.shape[0])
    col_actions=np.arange(col_matrix.shape[1])



    while True:
        dom_rows=find_strictly_dominated_actions(row_matrix)
        dom_cols=find_strictly_dominated_actions(col_matrix.T)

        if len(dom_rows) == 0 and len(dom_cols)==0:
            break

        row_matrix = np.delete(row_matrix, dom_rows, axis=0)
        col_matrix=np.delete(col_matrix, dom_rows, axis=0)

        row_matrix=np.delete(row_matrix, dom_cols, axis=1)
        col_matrix=np.delete(col_matrix, dom_cols, axis=1)

        row_actions=np.delete(row_actions, dom_rows)
        col_actions=np.delete(col_actions, dom_cols)
    return (row_matrix, col_matrix, row_actions, col_actions)

def main() -> None:
    pass


if __name__ == '__main__':
    main()
