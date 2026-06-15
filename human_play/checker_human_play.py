# -*- coding: utf-8 -*-
"""
Human VS AI for Checkers.
Input: from_dark,to_dark (0..31)  Example: 12,16
"""

from __future__ import print_function
from games.checker_game import Board, Game
from _mcts_alphazero.checker_mcts_alphaZero import MCTSPlayer
from policy_value_net.checker_policy_value_net_pytorch import PolicyValueNet


class Human(object):
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            s = input("Your move (from,to): ")
            location = [int(n, 10) for n in s.split(",")]
            move = board.location_to_move(location)
        except Exception:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            return self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    model_file = "best_checker_policy.model"
    board = Board(width=8, height=8, max_moves=200, promote_ends_turn=True)
    game = Game(board)

    best_policy = PolicyValueNet(
        8, 8,
        model_file=model_file,
        use_gpu=True,
        action_size=board.action_size,
        in_channels=6,
    )

    ai = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=80, is_selfplay=0)
    human = Human()
    game.start_play(human, ai, start_player=0, is_shown=1)


if __name__ == "__main__":
    run()
