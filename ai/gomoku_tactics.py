# -*- coding: utf-8 -*-
"""Small tactical layer for Gomoku move selection.

The neural net is still the main player. This module only handles direct
one-move tactics that a trained player should never miss: win now, block a
win, make/block fours, and make/block open threes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


@dataclass(frozen=True)
class MoveFeatures:
    win: bool = False
    open_four: int = 0
    closed_four: int = 0
    open_three: int = 0
    closed_three: int = 0
    open_two: int = 0

    @property
    def forcing_lines(self) -> int:
        return self.open_four + self.closed_four + self.open_three


def opponent(player: int) -> int:
    return 1 if int(player) == 2 else 2


def board_grid(board) -> np.ndarray:
    grid = np.zeros((board.height, board.width), dtype=np.int8)
    for move, player in board.states.items():
        move = int(move)
        grid[move // board.width, move % board.width] = int(player)
    return grid


def _inside(grid: np.ndarray, row: int, col: int) -> bool:
    return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]


def _count(grid: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> int:
    total = 0
    row += dr
    col += dc
    while _inside(grid, row, col) and grid[row, col] == player:
        total += 1
        row += dr
        col += dc
    return total


def _is_open(grid: np.ndarray, row: int, col: int) -> bool:
    return _inside(grid, row, col) and grid[row, col] == 0


def move_features(board, move: int, player: int, grid: Optional[np.ndarray] = None) -> MoveFeatures:
    if grid is None:
        grid = board_grid(board)

    move = int(move)
    player = int(player)
    row, col = move // board.width, move % board.width
    if not _inside(grid, row, col) or grid[row, col] != 0:
        return MoveFeatures()

    grid[row, col] = player
    try:
        n = int(board.n_in_row)
        open_four = 0
        closed_four = 0
        open_three = 0
        closed_three = 0
        open_two = 0

        for dr, dc in DIRECTIONS:
            left = _count(grid, row, col, -dr, -dc, player)
            right = _count(grid, row, col, dr, dc, player)
            total = 1 + left + right

            left_open = _is_open(grid, row - dr * (left + 1), col - dc * (left + 1))
            right_open = _is_open(grid, row + dr * (right + 1), col + dc * (right + 1))
            open_ends = int(left_open) + int(right_open)

            if total >= n:
                return MoveFeatures(win=True)
            if total == n - 1:
                if open_ends == 2:
                    open_four += 1
                elif open_ends == 1:
                    closed_four += 1
            elif total == n - 2:
                if open_ends == 2:
                    open_three += 1
                elif open_ends == 1:
                    closed_three += 1
            elif total == n - 3 and open_ends == 2:
                open_two += 1

        return MoveFeatures(
            open_four=open_four,
            closed_four=closed_four,
            open_three=open_three,
            closed_three=closed_three,
            open_two=open_two,
        )
    finally:
        grid[row, col] = 0


def tactical_score(features: MoveFeatures) -> float:
    if features.win:
        return 1_000_000.0

    score = 0.0
    score += features.open_four * 120_000.0
    score += features.closed_four * 45_000.0
    score += features.open_three * 9_000.0
    score += features.closed_three * 1_800.0
    score += features.open_two * 300.0

    if features.open_four + features.closed_four >= 2:
        score += 80_000.0
    if features.open_four + features.open_three >= 2:
        score += 35_000.0
    if features.open_three >= 2:
        score += 18_000.0
    return score


def positional_score(board, move: int, grid: Optional[np.ndarray] = None) -> float:
    if grid is None:
        grid = board_grid(board)

    row, col = int(move) // board.width, int(move) % board.width
    center_row = (board.height - 1) / 2.0
    center_col = (board.width - 1) / 2.0
    center = -((row - center_row) ** 2 + (col - center_col) ** 2) ** 0.5

    neighbors = 0
    for rr in range(max(0, row - 2), min(board.height, row + 3)):
        for cc in range(max(0, col - 2), min(board.width, col + 3)):
            if grid[rr, cc] != 0:
                neighbors += 1
    return center * 0.05 + neighbors * 0.2


def _best_by_position(board, moves: Iterable[int], grid: Optional[np.ndarray] = None) -> Optional[int]:
    moves = list(moves)
    if not moves:
        return None
    if grid is None:
        grid = board_grid(board)
    return int(max(moves, key=lambda move: (positional_score(board, move, grid), -int(move))))


def select_tactical_move(board, min_score: float = 8_000.0, mode: str = "full") -> Optional[int]:
    """Return a tactical move, or None when normal MCTS should decide."""
    mode = str(mode or "full").strip().lower()
    if mode in ("off", "none", "0", "false"):
        return None

    legal = list(getattr(board, "availables", []) or [])
    if not legal:
        return None

    player = int(board.current_player)
    other = opponent(player)
    grid = board_grid(board)

    own_wins = [move for move in legal if move_features(board, move, player, grid).win]
    if own_wins:
        return _best_by_position(board, own_wins, grid)

    opponent_wins = [move for move in legal if move_features(board, move, other, grid).win]
    if opponent_wins:
        return _best_by_position(board, opponent_wins, grid)

    if mode in ("emergency", "minimal"):
        return None

    best_move = None
    best_score = float("-inf")
    for move in legal:
        own_features = move_features(board, move, player, grid)
        opp_features = move_features(board, move, other, grid)

        own_score = tactical_score(own_features)
        block_score = tactical_score(opp_features) * 0.96
        score = max(own_score, block_score) + positional_score(board, move, grid)

        if score > best_score:
            best_score = score
            best_move = int(move)

    if best_move is not None and best_score >= min_score:
        return best_move
    return None
