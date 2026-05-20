# -*- coding: utf-8 -*-
"""
Pure MCTS baseline adapted for Checkers with possible non-switching moves.
"""

import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(board):
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    action_probs = np.ones(len(board.availables)) / max(1, len(board.availables))
    return zip(board.availables, action_probs), 0.0


class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0.0
        self._u = 0.0
        self._P = float(prior_p)

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = float(c_puct)
        self._n_playout = int(n_playout)

    def _evaluate_rollout(self, state, limit=200):
        player0 = state.get_current_player()
        for _ in range(limit):
            end, winner = state.game_end()
            if end:
                if winner == -1:
                    return 0.0, player0
                return (1.0 if winner == player0 else -1.0), player0
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        # move limit draw
        return 0.0, player0

    def _playout(self, state):
        node = self._root
        path_nodes = [node]
        path_players = [state.get_current_player()]

        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.do_move(action)
            path_nodes.append(node)
            path_players.append(state.get_current_player())

        action_probs, _ = self._policy(state)
        end, _winner = state.game_end()
        if not end:
            node.expand(action_probs)

        leaf_value, player_leaf = self._evaluate_rollout(state)

        for n, p in zip(reversed(path_nodes), reversed(path_players)):
            v = leaf_value if p == player_leaf else -leaf_value
            n.update(v)

    def get_move(self, state):
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: no legal moves")
            return -1

    def __str__(self):
        return "MCTS {}".format(self.player)