# ai/gomoku_az_agent.py
# -*- coding: utf-8 -*-
import os
try:
    from eventlet.semaphore import Semaphore
except ModuleNotFoundError:
    from threading import Lock

    def Semaphore(_value=1):
        return Lock()

from ai.gomoku_tactics import select_tactical_move
from policy_value_net.gomoku_policy_value_net_pytorch import PolicyValueNet
from _mcts_alphazero.mcts_alphaZero import MCTSPlayer


class GomokuPolicyPool:
    """全局只加载一次模型锁保护 policy_value_fn。"""
    def __init__(self, width=15, height=15, model_file="best_policy15.model", use_gpu=True, skip_attention=False):
        self.width = int(width)
        self.height = int(height)
        self.model_file = model_file
        self.use_gpu = bool(use_gpu)
        self.skip_attention = bool(skip_attention)

        self._lock = Semaphore(1)
        self.net = PolicyValueNet(
            self.width,
            self.height,
            model_file=self.model_file,
            use_gpu=self.use_gpu,
            skip_attention=self.skip_attention,
        )

    def policy_value_fn(self, board):
        with self._lock:
            return self.net.policy_value_fn(board)

    def policy_value_fn_batch(self, boards):
        with self._lock:
            return self.net.policy_value_fn_batch(boards)


class GomokuAZAgent:
    """每局一个 agent"""
    def __init__(self, pool: GomokuPolicyPool, c_puct=5, n_playout=80, mcts_batch_size=8, tactical_mode="full"):
        self.tactical_mode = str(tactical_mode or "full").strip().lower()
        self.player = MCTSPlayer(
            pool.policy_value_fn,
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=0,
            policy_value_batch_function=pool.policy_value_fn_batch,
            batch_size=mcts_batch_size,
        )

    def reset(self):
        #  MCTSPlayer 每次走完清
        try:
            self.player.reset_player()
        except Exception:
            pass
       
    def get_action(self, board):
        return self.select_move(board)

    
    def update_with_move(self, last_move: int):
        try:
            self.player.mcts.update_with_move(int(last_move))
        except Exception:
            pass
    def select_move(self, board):
        # 返回 action index
        if self.tactical_mode not in ("off", "none", "0", "false"):
            tactical_move = select_tactical_move(board, mode=self.tactical_mode)
            if tactical_move is not None:
                return int(tactical_move)

        mv = self.player.get_action(board, temp=1e-3, return_prob=0)
        return int(mv)
