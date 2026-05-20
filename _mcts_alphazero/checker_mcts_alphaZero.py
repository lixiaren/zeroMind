# -*- coding: utf-8 -*-
"""
AlphaZero-style MCTS adapted for Checkers 
"""

import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # action -> TreeNode
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
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000,
                 policy_value_batch_fn=None, batch_size=1):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._policy_batch = policy_value_batch_fn
        self._c_puct = float(c_puct)
        self._n_playout = int(n_playout)
        self._batch_size = max(1, int(batch_size))

    def _add_virtual_visits(self, path_nodes):
        for node in path_nodes:
            node._n_visits += 1

    def _revert_virtual_visits(self, path_nodes):
        for node in path_nodes:
            node._n_visits = max(0, node._n_visits - 1)

    def _select_leaf(self, state):
        node = self._root
        path_nodes = [node]
        path_players = [state.get_current_player()]

        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.do_move(action)
            path_nodes.append(node)
            path_players.append(state.get_current_player())

        end, winner = state.game_end()
        return {
            "node": node,
            "state": state,
            "path_nodes": path_nodes,
            "path_players": path_players,
            "end": end,
            "winner": winner,
            "player_leaf": state.get_current_player(),
        }

    def _finish_leaf(self, leaf, action_probs=None, leaf_value=None):
        self._revert_virtual_visits(leaf["path_nodes"])
        node = leaf["node"]
        player_leaf = leaf["player_leaf"]

        if not leaf["end"]:
            node.expand(action_probs)
            leaf_value = float(leaf_value)
        else:
            winner = leaf["winner"]
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == player_leaf else -1.0

        for n, p in zip(reversed(leaf["path_nodes"]), reversed(leaf["path_players"])):
            v = leaf_value if p == player_leaf else -leaf_value
            n.update(v)

    def _playout_batch(self, state, batch_size):
        leaves = []
        for _ in range(batch_size):
            state_copy = copy.deepcopy(state)
            leaf = self._select_leaf(state_copy)
            self._add_virtual_visits(leaf["path_nodes"])
            leaves.append(leaf)

        non_terminal = [leaf for leaf in leaves if not leaf["end"]]
        batch_results = []
        if non_terminal:
            if self._policy_batch is not None:
                batch_results = self._policy_batch([leaf["state"] for leaf in non_terminal])
            else:
                batch_results = [self._policy(leaf["state"]) for leaf in non_terminal]

        result_i = 0
        for leaf in leaves:
            if leaf["end"]:
                self._finish_leaf(leaf)
            else:
                action_probs, leaf_value = batch_results[result_i]
                result_i += 1
                self._finish_leaf(leaf, action_probs, leaf_value)

    def _playout(self, state):
        node = self._root

        path_nodes = [node]
        path_players = [state.get_current_player()]  # player-to-play at each node

        # selection
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.do_move(action)
            path_nodes.append(node)
            path_players.append(state.get_current_player())

        # evaluate leaf
        action_probs, leaf_value = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
            player_leaf = state.get_current_player()
            leaf_value = float(leaf_value)
        else:
            player_leaf = state.get_current_player()
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == player_leaf else -1.0

        # backprop: value from each node's player perspective
        for n, p in zip(reversed(path_nodes), reversed(path_players)):
            v = leaf_value if p == player_leaf else -leaf_value
            n.update(v)

    def get_move_probs(self, state, temp=1e-3):
        remaining = int(self._n_playout)
        if remaining > 0 and self._root.is_leaf():
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
            remaining -= 1

        if self._policy_batch is None or self._batch_size <= 1:
            for _ in range(remaining):
                state_copy = copy.deepcopy(state)
                self._playout(state_copy)
        else:
            while remaining > 0:
                current_batch = min(self._batch_size, remaining)
                self._playout_batch(state, current_batch)
                remaining -= current_batch

        act_visits = [(act, n._n_visits) for act, n in self._root._children.items()]
        if not act_visits:
            acts = tuple(state.availables)
            probs = np.ones(len(acts), dtype=np.float32) / len(acts)
            return acts, probs
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0,
                 policy_value_batch_function=None, batch_size=1):
        self.mcts = MCTS(
            policy_value_function,
            c_puct,
            n_playout,
            policy_value_batch_fn=policy_value_batch_function,
            batch_size=batch_size,
        )
        self._is_selfplay = int(is_selfplay)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        action_size = int(getattr(board, "action_size", board.width * board.height))
        move_probs = np.zeros(action_size, dtype=np.float32)

        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs

            if self._is_selfplay:
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            return move
        else:
            print("WARNING: no legal moves")
            return -1 if not return_prob else (-1, move_probs)

    def __str__(self):
        return "MCTS {}".format(self.player)
