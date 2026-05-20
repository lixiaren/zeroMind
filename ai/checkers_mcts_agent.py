# -*- coding: utf-8 -*-
import threading
from ai.checkers_tactics import (
    action_score,
    is_no_benefit_sacrifice,
    select_fallback_action,
    select_tactical_action,
)
from policy_value_net.checker_policy_value_net_pytorch import PolicyValueNet
from _mcts_alphazero.checker_mcts_alphaZero import MCTSPlayer


class CheckersPolicyPool:
    """模型只加载一次 多个对局复用 简单加锁防并发推理冲突"""
    def __init__(self, model_file: str = "best_checker_policy.model", use_gpu: bool = True, skip_attention: bool = False):
        self.skip_attention = bool(skip_attention)
        self.net = PolicyValueNet(
            8, 8,
            model_file=model_file,
            use_gpu=use_gpu,
            action_size=32 * 32,
            in_channels=6, 
            skip_attention=self.skip_attention,
        )
        self._lock = threading.Lock()

    def policy_value_fn(self, board):
        with self._lock:
            return self.net.policy_value_fn(board)

    def policy_value_fn_batch(self, boards):
        with self._lock:
            return self.net.policy_value_fn_batch(boards)


class CheckersAZAgent:
    def __init__(
        self,
        pool: CheckersPolicyPool,
        c_puct: float = 5.0,
        n_playout: int = 80,
        mcts_batch_size: int = 8,
        safety_gap: float = 60.0,
        tactical_mode: str = "full",
    ):
        self.safety_gap = float(safety_gap)
        self.tactical_mode = str(tactical_mode or "full").strip().lower()
        self.player = MCTSPlayer(
            pool.policy_value_fn,
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=0,
            policy_value_batch_function=pool.policy_value_fn_batch,
            batch_size=mcts_batch_size,
        )

    def get_action(self, board):
        if self.tactical_mode == "full":
            tactical_action = select_tactical_action(board)
            if tactical_action is not None:
                return int(tactical_action)

        try:
            action = int(self.player.get_action(board, temp=1e-3))
        except Exception:
            action = -1

        if action not in list(board.availables or []):
            fallback = select_fallback_action(board)
            if fallback is not None:
                return int(fallback)

        if self.tactical_mode in ("off", "none", "0", "false"):
            return int(action)

        fallback = select_fallback_action(board)
        if fallback is not None and fallback != action:
            try:
                if is_no_benefit_sacrifice(board, action):
                    return int(fallback)
                if action_score(board, fallback) - action_score(board, action) >= self.safety_gap:
                    return int(fallback)
            except Exception:
                pass
        return int(action)
