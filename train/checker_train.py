# -*- coding: utf-8 -*-
"""AlphaZero-style training pipeline for checkers.

Default profile is shortened for CPU training. Use --profile classic for the
old long-running settings.
"""

from __future__ import print_function

import argparse
import copy
import json
import os
import pickle
import random
import sys
from collections import defaultdict, deque

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from games.checker_game import Board, Game, EMPTY, P1_MAN, P2_MAN, P1_KING, P2_KING
from ai.checkers_tactics import action_outcome, action_score, is_no_benefit_sacrifice
from pure_mcts.checker_mcts_pure import MCTSPlayer as MCTS_Pure
from _mcts_alphazero.checker_mcts_alphaZero import MCTSPlayer
from policy_value_net.checker_policy_value_net_pytorch import PolicyValueNet


CHECKERS_PROFILES = {
    "day": {
        "learn_rate": 8e-4,
        "temp": 1.0,
        "n_playout": 96,
        "c_puct": 4.0,
        "buffer_size": 12000,
        "batch_size": 256,
        "min_data_for_train": 512,
        "train_freq": 1,
        "play_batch_size": 1,
        "epochs": 2,
        "kl_targ": 0.02,
        "check_freq": 25,
        "game_batch_num": 700,
        "eval_games": 4,
        "pure_mcts_playout_num": 800,
        "pure_inc_threshold": 0.75,
        "pure_inc_step": 200,
        "pure_max": 3000,
        "max_moves": 160,
        "channels": 128,
        "n_blocks": 10,
        "use_attention": True,
        "tactic_ratio": 0.20,
    },
    "balanced": {
        "learn_rate": 8e-4,
        "temp": 1.0,
        "n_playout": 180,
        "c_puct": 4.5,
        "buffer_size": 18000,
        "batch_size": 256,
        "min_data_for_train": 1000,
        "train_freq": 1,
        "play_batch_size": 1,
        "epochs": 3,
        "kl_targ": 0.02,
        "check_freq": 40,
        "game_batch_num": 1300,
        "eval_games": 6,
        "pure_mcts_playout_num": 1500,
        "pure_inc_threshold": 0.80,
        "pure_inc_step": 250,
        "pure_max": 6000,
        "max_moves": 180,
        "channels": 128,
        "n_blocks": 10,
        "use_attention": True,
        "tactic_ratio": 0.20,
    },
    "classic": {
        "learn_rate": 1e-3,
        "temp": 1.0,
        "n_playout": 400,
        "c_puct": 5.0,
        "buffer_size": 20000,
        "batch_size": 256,
        "min_data_for_train": 2000,
        "train_freq": 1,
        "play_batch_size": 1,
        "epochs": 5,
        "kl_targ": 0.02,
        "check_freq": 50,
        "game_batch_num": 2000,
        "eval_games": 10,
        "pure_mcts_playout_num": 5000,
        "pure_inc_threshold": 0.80,
        "pure_inc_step": 200,
        "pure_max": 10000,
        "max_moves": 200,
        "channels": 128,
        "n_blocks": 10,
        "use_attention": True,
        "tactic_ratio": 0.20,
    },
}


def build_milestone_checkpoints(spec, prefix):
    spec = str(spec or "").strip()
    if not spec or spec.lower() in ("none", "off", "0"):
        return {}

    checkpoints = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        # Accept both "1000" and "1000:normal"; the label is only for humans.
        batch_text = item.split(":", 1)[0].strip()
        batch = int(batch_text)
        checkpoints[batch] = "{}_{:04d}.model".format(prefix, batch)
    return checkpoints


def build_playout_schedule(spec):
    spec = str(spec or "").strip()
    if not spec or spec.lower() in ("none", "off", "0"):
        return []

    schedule = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        batch_text, playout_text = item.split(":", 1)
        schedule.append((int(batch_text.strip()), int(playout_text.strip())))
    return sorted(schedule)


def load_train_state(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TrainPipeline(object):
    def __init__(self, init_model=None, profile="day", use_gpu=True, **overrides):
        if profile not in CHECKERS_PROFILES:
            raise ValueError("unknown profile: {}".format(profile))

        cfg = dict(CHECKERS_PROFILES[profile])
        cfg.update({key: value for key, value in overrides.items() if value is not None})

        self.profile = profile
        self.board_width = 8
        self.board_height = 8
        self.board = Board(
            width=8,
            height=8,
            max_moves=int(cfg["max_moves"]),
            promote_ends_turn=True,
        )
        self.game = Game(self.board)

        self.learn_rate = float(cfg["learn_rate"])
        self.lr_multiplier = float(cfg.get("lr_multiplier", 1.0))
        self.temp = float(cfg["temp"])
        self.n_playout = int(cfg["n_playout"])
        self.c_puct = float(cfg["c_puct"])
        self.buffer_size = int(cfg["buffer_size"])
        self.batch_size = int(cfg["batch_size"])
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.min_data_for_train = int(cfg["min_data_for_train"])
        self.train_freq = int(cfg["train_freq"])
        self.play_batch_size = int(cfg["play_batch_size"])
        self.epochs = int(cfg["epochs"])
        self.kl_targ = float(cfg["kl_targ"])
        self.check_freq = int(cfg["check_freq"])
        self.game_batch_num = int(cfg["game_batch_num"])
        self.eval_games = int(cfg["eval_games"])
        self.best_win_ratio = float(cfg.get("best_win_ratio", 0.0))
        self.pure_mcts_playout_num = int(cfg["pure_mcts_playout_num"])
        self.pure_inc_threshold = float(cfg["pure_inc_threshold"])
        self.pure_inc_step = int(cfg["pure_inc_step"])
        self.pure_max = int(cfg["pure_max"])
        self.current_model_file = cfg.get("current_model_file", "./current_checker_policy.model")
        self.best_model_file = cfg.get("best_model_file", "./best_checker_policy.model")
        self.milestone_checkpoints = dict(cfg.get("milestone_checkpoints", {}))
        self.playout_schedule = list(cfg.get("playout_schedule", []))
        self.start_batch = int(cfg.get("start_batch", 0))
        self.state_file = cfg.get("state_file", "./checker_train_state.json")
        self.use_gpu = bool(use_gpu)
        self.channels = int(cfg["channels"])
        self.n_blocks = int(cfg["n_blocks"])
        self.use_attention = bool(cfg["use_attention"])
        self.use_amp = bool(cfg.get("use_amp", False))
        self.channels_last = bool(cfg.get("channels_last", False))
        self.mcts_batch_size = int(cfg.get("mcts_batch_size", 8))
        self.tactic_ratio = max(0.0, min(0.5, float(cfg.get("tactic_ratio", 0.0))))
        self.buffer_file = cfg.get("buffer_file", "./checker_data_buffer.pkl")
        self.buffer_save_freq = int(cfg.get("buffer_save_freq", 20))
        self.restore_buffer = bool(cfg.get("restore_buffer", False))

        self._mirror_dark = self._build_mirror_dark()
        self._mirror_action = self._build_mirror_action()

        self.policy_value_net = PolicyValueNet(
            self.board_width,
            self.board_height,
            model_file=init_model,
            use_gpu=self.use_gpu,
            action_size=self.board.action_size,
            in_channels=6,
            channels=self.channels,
            n_blocks=self.n_blocks,
            use_attention=self.use_attention,
            use_amp=self.use_amp,
            channels_last=self.channels_last,
        )
        loaded_net = self.policy_value_net.policy_value_net
        self.channels = int(loaded_net.channels)
        self.n_blocks = int(loaded_net.n_blocks)
        self.use_attention = bool(loaded_net.use_attention)
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1,
            policy_value_batch_function=self.policy_value_net.policy_value_fn_batch,
            batch_size=self.mcts_batch_size,
        )
        self._schedule_idx = 0
        if self.restore_buffer:
            self._load_data_buffer()

    def _apply_playout_schedule(self, batch_i):
        changed = False
        while self._schedule_idx < len(self.playout_schedule):
            start_batch, playouts = self.playout_schedule[self._schedule_idx]
            if batch_i < start_batch:
                break
            self.n_playout = int(playouts)
            self.mcts_player.mcts._n_playout = self.n_playout
            self._schedule_idx += 1
            changed = True
        if changed:
            print("Update self-play n_playout -> {} at batch {}".format(self.n_playout, batch_i))

    def _save_train_state(self, batch_i):
        if not self.state_file:
            return
        state = {
            "completed_batches": int(batch_i),
            "profile": self.profile,
            "board_width": self.board_width,
            "board_height": self.board_height,
            "learn_rate": float(self.learn_rate),
            "lr_multiplier": float(self.lr_multiplier),
            "temperature": float(self.temp),
            "c_puct": float(self.c_puct),
            "n_playout": int(self.n_playout),
            "mcts_batch_size": int(self.mcts_batch_size),
            "playout_schedule": list(self.playout_schedule),
            "buffer_size": int(self.buffer_size),
            "batch_size": int(self.batch_size),
            "min_data_for_train": int(self.min_data_for_train),
            "train_freq": int(self.train_freq),
            "play_batch_size": int(self.play_batch_size),
            "epochs": int(self.epochs),
            "kl_targ": float(self.kl_targ),
            "check_freq": int(self.check_freq),
            "eval_games": int(self.eval_games),
            "current_model_file": self.current_model_file,
            "best_model_file": self.best_model_file,
            "pure_mcts_playout_num": int(self.pure_mcts_playout_num),
            "pure_inc_threshold": float(self.pure_inc_threshold),
            "pure_inc_step": int(self.pure_inc_step),
            "pure_max": int(self.pure_max),
            "best_win_ratio": float(self.best_win_ratio),
            "max_moves": int(self.board.max_moves),
            "channels": int(self.channels),
            "n_blocks": int(self.n_blocks),
            "use_attention": bool(self.use_attention),
            "tactic_ratio": float(self.tactic_ratio),
            "buffer_file": self.buffer_file,
            "buffer_save_freq": int(self.buffer_save_freq),
            "buffer_len": int(len(self.data_buffer)),
        }
        tmp_file = self.state_file + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp_file, self.state_file)

    def _save_data_buffer(self):
        if not self.buffer_file:
            return
        directory = os.path.dirname(os.path.abspath(self.buffer_file))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "version": 1,
            "game": "checkers",
            "board_width": int(self.board_width),
            "board_height": int(self.board_height),
            "max_moves": int(self.board.max_moves),
            "buffer_size": int(self.buffer_size),
            "data": list(self.data_buffer),
        }
        tmp_file = self.buffer_file + ".tmp"
        with open(tmp_file, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_file, self.buffer_file)
        print("Save data_buffer: {} samples -> {}".format(len(self.data_buffer), self.buffer_file))

    def _load_data_buffer(self):
        if not self.buffer_file:
            return
        if not os.path.exists(self.buffer_file):
            print("No buffer file found; start with empty data_buffer:", self.buffer_file)
            return
        with open(self.buffer_file, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get("game") != "checkers":
            print("Skip incompatible buffer file:", self.buffer_file)
            return
        if int(payload.get("board_width", 0)) != self.board_width or int(payload.get("board_height", 0)) != self.board_height:
            print("Skip buffer with different board size:", self.buffer_file)
            return
        data = list(payload.get("data") or [])
        if len(data) > self.buffer_size:
            data = data[-self.buffer_size:]
        self.data_buffer = deque(data, maxlen=self.buffer_size)
        print("Restore data_buffer: {} samples <- {}".format(len(self.data_buffer), self.buffer_file))

    def _build_mirror_dark(self):
        mirror = np.zeros(32, dtype=np.int32)
        for di in range(32):
            row, col = self.board._dark_to_rc[di]
            col4 = col // 2
            mirrored_col4 = 3 - col4
            mirrored_col = 2 * mirrored_col4 + (1 if row % 2 == 0 else 0)
            mirror[di] = int(self.board._rc_to_dark[row, mirrored_col])
        return mirror

    def _build_mirror_action(self):
        action_map = np.zeros(self.board.action_size, dtype=np.int32)
        for action in range(self.board.action_size):
            from_idx, to_idx = action // 32, action % 32
            action_map[action] = int(self._mirror_dark[from_idx]) * 32 + int(self._mirror_dark[to_idx])
        return action_map

    def _mirror_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (6, 8, 8):
            raise ValueError("unexpected state shape: {}".format(state.shape))

        mirrored = np.zeros_like(state, dtype=np.float32)
        mirrored[4, :, :] = state[4, :, :]
        for dark_idx in range(32):
            row, col = self.board._dark_to_rc[dark_idx]
            mirrored_dark = int(self._mirror_dark[dark_idx])
            mirrored_row, mirrored_col = self.board._dark_to_rc[mirrored_dark]
            for plane in (0, 1, 2, 3, 5):
                mirrored[plane, mirrored_row, mirrored_col] = state[plane, row, col]
        return mirrored

    def get_equi_data(self, play_data):
        extend = []
        for state, mcts_prob, winner in play_data:
            extend.append((state, mcts_prob, winner))

            mirrored_state = self._mirror_state(state)
            prob = np.asarray(mcts_prob, dtype=np.float32)
            mirrored_prob = np.zeros_like(prob)
            mirrored_prob[self._mirror_action] = prob
            extend.append((mirrored_state, mirrored_prob, winner))
        return extend

    def _new_empty_tactic_board(self, player):
        board = Board(width=8, height=8, max_moves=int(self.board.max_moves), promote_ends_turn=True)
        board.board[:, :] = EMPTY
        board.current_player = int(player)
        board.last_move = -1
        board.move_count = random.randint(8, 40)
        board._chain_pos = None
        board.states = {}
        board.availables = []
        return board

    def _checker_prob(self, actions):
        probs = np.zeros(self.board.action_size, dtype=np.float32)
        actions = list(actions)
        if not actions:
            return probs
        p = 1.0 / float(len(actions))
        for action in actions:
            probs[int(action)] = p
        return probs

    def _checker_ranked_prob(self, scored_actions, floor=0.03, power=2.4):
        """Soft-label legal moves by tactical score without making bad moves impossible."""
        probs = np.zeros(self.board.action_size, dtype=np.float32)
        scored_actions = [(int(a), float(s)) for a, s in scored_actions]
        if not scored_actions:
            return probs
        if len(scored_actions) == 1:
            probs[scored_actions[0][0]] = 1.0
            return probs

        scores = np.asarray([s for _, s in scored_actions], dtype=np.float32)
        span = float(np.max(scores) - np.min(scores))
        if span <= 1e-6:
            return self._checker_prob([a for a, _ in scored_actions])

        scaled = (scores - float(np.min(scores))) / span
        weights = np.maximum(float(floor), np.power(scaled, float(power)))
        weights = weights / float(np.sum(weights))
        for (action, _), weight in zip(scored_actions, weights):
            probs[int(action)] = float(weight)
        return probs

    def _score_checker_actions(self, board, actions):
        scored = []
        for action in actions:
            try:
                score = float(action_score(board, int(action)))
                gain, reply_cost, has_capture_reply = action_outcome(board, int(action))
                net = float(gain - reply_cost)
                score += net * 3.0
                if has_capture_reply and net < -15.0:
                    score -= 350.0
                scored.append((int(action), score, net, bool(has_capture_reply)))
            except Exception:
                scored.append((int(action), -9999.0, -9999.0, True))
        return scored

    def _is_capture_action(self, board, action):
        from_idx = int(action) // 32
        to_idx = int(action) % 32
        fr, fc = board._dark_to_rc[from_idx]
        tr, tc = board._dark_to_rc[to_idx]
        return abs(tr - fr) == 2 and abs(tc - fc) == 2

    def _make_capture_tactic_sample(self, chain=False):
        for _ in range(200):
            player = random.choice((1, 2))
            board = self._new_empty_tactic_board(player)
            current_man = P1_MAN if player == 1 else P2_MAN
            current_king = P1_KING if player == 1 else P2_KING
            opp_man = P2_MAN if player == 1 else P1_MAN
            opp_king = P2_KING if player == 1 else P1_KING
            piece = current_king if random.random() < 0.35 else current_man
            dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if piece in (P1_KING, P2_KING) else (
                [(1, -1), (1, 1)] if player == 1 else [(-1, -1), (-1, 1)]
            )
            dr, dc = random.choice(dirs)
            from_idx = random.randrange(32)
            r, c = board._dark_to_rc[from_idx]
            mr, mc = r + dr, c + dc
            tr, tc = r + 2 * dr, c + 2 * dc
            if not (0 <= mr < 8 and 0 <= mc < 8 and 0 <= tr < 8 and 0 <= tc < 8):
                continue
            if board._rc_to_dark[tr, tc] < 0:
                continue
            board.board[r, c] = piece
            board.board[mr, mc] = opp_king if random.random() < 0.2 else opp_man
            if chain:
                board._chain_pos = (r, c)
            board.availables = board.get_legal_actions()
            captures = [a for a in board.availables if self._is_capture_action(board, a)]
            if not captures:
                continue
            value = 0.75 if chain else 0.55
            return board.current_state(), self._checker_prob(captures), value
        return self._make_promotion_tactic_sample()

    def _make_promotion_tactic_sample(self):
        for _ in range(200):
            player = random.choice((1, 2))
            board = self._new_empty_tactic_board(player)
            piece = P1_MAN if player == 1 else P2_MAN
            row = 6 if player == 1 else 1
            from_candidates = [di for di, (r, c) in enumerate(board._dark_to_rc) if r == row]
            random.shuffle(from_candidates)
            for from_idx in from_candidates:
                r, c = board._dark_to_rc[from_idx]
                board.board[:, :] = EMPTY
                board.board[r, c] = piece
                board._chain_pos = None
                board.availables = board.get_legal_actions()
                promotion_actions = []
                for action in board.availables:
                    to_idx = int(action) % 32
                    tr, tc = board._dark_to_rc[to_idx]
                    if (player == 1 and tr == 7) or (player == 2 and tr == 0):
                        promotion_actions.append(action)
                if promotion_actions:
                    return board.current_state(), self._checker_prob(promotion_actions), 0.45
        return self._make_capture_tactic_sample(chain=False)

    def _action_loses_moved_piece_immediately(self, board, action):
        if self._is_capture_action(board, action):
            return False
        to_idx = int(action) % 32
        tr, tc = board._dark_to_rc[to_idx]
        try:
            next_board = copy.deepcopy(board)
            next_board.do_move(int(action))
        except Exception:
            return True
        for reply in next_board.availables:
            if not self._is_capture_action(next_board, reply):
                continue
            rf = int(reply) // 32
            rt = int(reply) % 32
            rr, rc = next_board._dark_to_rc[rf]
            nr, nc = next_board._dark_to_rc[rt]
            jumped = ((rr + nr) // 2, (rc + nc) // 2)
            if jumped == (tr, tc):
                return True
        return False

    def _random_small_checker_position(self, player, own_range=(2, 4), opp_range=(2, 4), king_prob=0.25):
        board = self._new_empty_tactic_board(player)
        dark_indices = list(range(32))
        random.shuffle(dark_indices)
        occupied = set()
        own_count = random.randint(int(own_range[0]), int(own_range[1]))
        opp_count = random.randint(int(opp_range[0]), int(opp_range[1]))
        for _ in range(own_count):
            if not dark_indices:
                break
            di = dark_indices.pop()
            occupied.add(di)
            r, c = board._dark_to_rc[di]
            if player == 1:
                board.board[r, c] = P1_KING if random.random() < king_prob else P1_MAN
            else:
                board.board[r, c] = P2_KING if random.random() < king_prob else P2_MAN
        for _ in range(opp_count):
            if not dark_indices:
                break
            di = dark_indices.pop()
            if di in occupied:
                continue
            r, c = board._dark_to_rc[di]
            if player == 1:
                board.board[r, c] = P2_KING if random.random() < king_prob else P2_MAN
            else:
                board.board[r, c] = P1_KING if random.random() < king_prob else P1_MAN
        board.availables = board.get_legal_actions()
        return board

    def _make_capture_choice_tactic_sample(self):
        for _ in range(900):
            player = random.choice((1, 2))
            board = self._random_small_checker_position(
                player,
                own_range=(3, 6),
                opp_range=(3, 6),
                king_prob=0.22,
            )
            captures = [a for a in board.availables if self._is_capture_action(board, a)]
            if len(captures) < 2:
                continue

            scored = self._score_checker_actions(board, captures)
            best_score = max(score for _, score, _, _ in scored)
            worst_score = min(score for _, score, _, _ in scored)
            best_net = max(net for _, _, net, _ in scored)
            worst_net = min(net for _, _, net, _ in scored)
            has_bad_reply = any(has_reply and net < -15.0 for _, _, net, has_reply in scored)
            if best_score - worst_score < 180.0 and best_net - worst_net < 120.0 and not has_bad_reply:
                continue

            probs = self._checker_ranked_prob([(a, s) for a, s, _, _ in scored], floor=0.015, power=2.8)
            value = float(np.clip(best_net / 300.0, -0.45, 0.75))
            return board.current_state(), probs, value
        return self._make_safety_tactic_sample()

    def _make_anti_blunder_tactic_sample(self):
        for _ in range(900):
            player = random.choice((1, 2))
            board = self._random_small_checker_position(
                player,
                own_range=(3, 6),
                opp_range=(3, 6),
                king_prob=0.18,
            )
            if not board.availables or any(self._is_capture_action(board, a) for a in board.availables):
                continue

            scored = self._score_checker_actions(board, board.availables)
            safe = [
                (a, score)
                for a, score, net, has_reply in scored
                if not has_reply and not is_no_benefit_sacrifice(board, a)
            ]
            unsafe = [
                (a, score)
                for a, score, net, has_reply in scored
                if has_reply or is_no_benefit_sacrifice(board, a)
            ]
            if not safe or not unsafe:
                continue

            probs = self._checker_ranked_prob([(a, s) for a, s, _, _ in scored], floor=0.01, power=3.0)
            return board.current_state(), probs, 0.20
        return self._make_safety_tactic_sample()

    def _make_safety_tactic_sample(self):
        for _ in range(500):
            player = random.choice((1, 2))
            board = self._random_small_checker_position(player)
            if not board.availables or any(self._is_capture_action(board, a) for a in board.availables):
                continue
            safe = []
            unsafe = []
            for action in board.availables:
                if (
                    self._action_loses_moved_piece_immediately(board, action)
                    or is_no_benefit_sacrifice(board, action)
                ):
                    unsafe.append(action)
                else:
                    safe.append(action)
            if not safe or not unsafe:
                continue
            scored = self._score_checker_actions(board, board.availables)
            probs = self._checker_ranked_prob([(a, s) for a, s, _, _ in scored], floor=0.02, power=2.8)
            return board.current_state(), probs, 0.25
        return self._make_capture_tactic_sample(chain=False)

    def _make_king_chase_tactic_sample(self):
        for _ in range(300):
            player = random.choice((1, 2))
            board = self._new_empty_tactic_board(player)
            own_king = P1_KING if player == 1 else P2_KING
            opp_piece = P2_MAN if player == 1 else P1_MAN
            king_idx = random.randrange(32)
            opp_idx = random.randrange(32)
            if king_idx == opp_idx:
                continue
            kr, kc = board._dark_to_rc[king_idx]
            orow, ocol = board._dark_to_rc[opp_idx]
            board.board[kr, kc] = own_king
            board.board[orow, ocol] = opp_piece
            board.availables = board.get_legal_actions()
            if not board.availables or any(self._is_capture_action(board, a) for a in board.availables):
                continue
            scores = []
            for action in board.availables:
                to_idx = int(action) % 32
                tr, tc = board._dark_to_rc[to_idx]
                before = abs(kr - orow) + abs(kc - ocol)
                after = abs(tr - orow) + abs(tc - ocol)
                improvement = max(0, before - after)
                tactical_score = float(action_score(board, int(action))) + 70.0 * float(improvement)
                scores.append((action, tactical_score))
            if len(scores) < 2 or max(s for _, s in scores) <= min(s for _, s in scores):
                continue
            probs = self._checker_ranked_prob(scores, floor=0.02, power=2.5)
            return board.current_state(), probs, 0.30
        return self._make_safety_tactic_sample()

    def _make_checker_tactic_sample(self):
        roll = random.random()
        if roll < 0.26:
            return self._make_capture_choice_tactic_sample()
        if roll < 0.46:
            return self._make_anti_blunder_tactic_sample()
        if roll < 0.62:
            return self._make_capture_tactic_sample(chain=False)
        if roll < 0.74:
            return self._make_capture_tactic_sample(chain=True)
        if roll < 0.84:
            return self._make_promotion_tactic_sample()
        if roll < 0.94:
            return self._make_safety_tactic_sample()
        return self._make_king_chase_tactic_sample()

    def _generate_tactic_batch(self, count):
        return [self._make_checker_tactic_sample() for _ in range(max(0, int(count)))]

    def collect_selfplay_data(self, n_games=1):
        for _ in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)
            self.episode_len = len(play_data)
            self.data_buffer.extend(self.get_equi_data(play_data))

    def policy_update(self):
        tactic_count = int(round(self.batch_size * self.tactic_ratio)) if self.tactic_ratio > 0 else 0
        tactic_count = max(0, min(self.batch_size - 1, tactic_count))
        selfplay_count = self.batch_size - tactic_count
        mini_batch = random.sample(self.data_buffer, selfplay_count)
        if tactic_count > 0:
            mini_batch.extend(self._generate_tactic_batch(tactic_count))
            random.shuffle(mini_batch)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        kl = 0.0
        for _ in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier,
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(
                np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1)
            )
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        winner_arr = np.array(winner_batch)
        old_var = np.var(winner_arr - old_v.flatten()) / (np.var(winner_arr) + 1e-10)
        new_var = np.var(winner_arr - new_v.flatten()) / (np.var(winner_arr) + 1e-10)
        print(
            "kl:{:.5f}, lr_mult:{:.3f}, loss:{:.4f}, entropy:{:.4f}, EV_old:{:.3f}, EV_new:{:.3f}".format(
                kl,
                self.lr_multiplier,
                loss,
                entropy,
                1 - old_var,
                1 - new_var,
            )
        )
        return loss, entropy

    def policy_evaluate(self, n_games=None):
        n_games = self.eval_games if n_games is None else int(n_games)
        current_mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            policy_value_batch_function=self.policy_value_net.policy_value_fn_batch,
            batch_size=self.mcts_batch_size,
        )
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(
                current_mcts_player,
                pure_mcts_player,
                start_player=i % 2,
                is_shown=0,
            )
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print(
            "pure_playouts:{}, win:{}, lose:{}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1],
                win_cnt[2],
                win_cnt[-1],
            )
        )
        return win_ratio

    def run(self):
        print(
                "profile:{}, start_batch:{}, n_playout:{}, mcts_batch:{}, games:{}, batch:{}, tactic_ratio:{:.2f}, min_data:{}, update_epochs:{}, eval_games:{}, net:{}x{}, attention:{}".format(
                self.profile,
                self.start_batch,
                self.n_playout,
                self.mcts_batch_size,
                self.game_batch_num,
                self.batch_size,
                self.tactic_ratio,
                self.min_data_for_train,
                self.epochs,
                self.eval_games,
                self.channels,
                self.n_blocks,
                self.use_attention,
            )
        )
        print("device:{}".format(self.policy_value_net.device))
        print(
            "buffer_file:{}, buffer_save_freq:{}, buffer_len:{}".format(
                self.buffer_file,
                self.buffer_save_freq,
                len(self.data_buffer),
            )
        )
        last_batch = self.start_batch
        try:
            for i in range(self.game_batch_num):
                batch_i = self.start_batch + i + 1
                last_batch = batch_i
                self._apply_playout_schedule(batch_i)
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}, buffer:{}".format(batch_i, self.episode_len, len(self.data_buffer)))

                if len(self.data_buffer) >= self.min_data_for_train and batch_i % self.train_freq == 0:
                    self.policy_update()

                if batch_i in self.milestone_checkpoints:
                    checkpoint_file = self.milestone_checkpoints[batch_i]
                    self.policy_value_net.save_model(checkpoint_file)
                    print("Save milestone checkpoint: batch {} -> {}".format(batch_i, checkpoint_file))

                if batch_i % self.check_freq == 0:
                    print("current self-play batch:", batch_i)
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model(self.current_model_file)

                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(self.best_model_file)

                    if win_ratio >= self.pure_inc_threshold and self.pure_mcts_playout_num < self.pure_max:
                        self.pure_mcts_playout_num = min(
                            self.pure_mcts_playout_num + self.pure_inc_step,
                            self.pure_max,
                        )
                        print("Increase pure_mcts_playout_num ->", self.pure_mcts_playout_num)
                        self.best_win_ratio = 0.0
                self._save_train_state(batch_i)
                if self.buffer_save_freq > 0 and batch_i % self.buffer_save_freq == 0:
                    self._save_data_buffer()
            if last_batch > self.start_batch:
                self._save_data_buffer()
                self._save_train_state(last_batch)

        except KeyboardInterrupt:
            if last_batch >= self.start_batch:
                print("\nSave current model before quit -> {}".format(self.current_model_file))
                self.policy_value_net.save_model(self.current_model_file)
                self._save_data_buffer()
                self._save_train_state(last_batch)
            print("\n\rquit")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train checkers policy-value network.")
    parser.add_argument("--profile", choices=sorted(CHECKERS_PROFILES), default="day")
    parser.add_argument("--init-model", default=None, help="Resume from an existing .model file.")
    parser.add_argument("--continue-last", action="store_true", help="Continue from the state recorded in --state-file.")
    parser.add_argument("--state-file", default="./checker_train_state.json", help="JSON file used to record resume metadata.")
    parser.add_argument("--buffer-file", default=None, help="Pickle file used to save and restore the self-play replay buffer.")
    parser.add_argument("--buffer-save-freq", type=int, default=None, help="Save the replay buffer every N completed self-play batches. Use 0 to only save on Ctrl+C/end.")
    parser.add_argument("--no-restore-buffer", action="store_true", help="With --continue-last, skip loading the replay buffer file.")
    parser.add_argument("--current-model", default="./current_checker_policy.model")
    parser.add_argument("--best-model", default="./best_checker_policy.model")
    parser.add_argument("--learn-rate", type=float, default=None)
    parser.add_argument("--lr-multiplier", type=float, default=None)
    parser.add_argument("--temperature", "--temp", dest="temp", type=float, default=None)
    parser.add_argument("--game-batches", type=int, default=None)
    parser.add_argument("--playouts", type=int, default=None)
    parser.add_argument("--c-puct", type=float, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-data-for-train", type=int, default=None)
    parser.add_argument("--train-freq", type=int, default=None)
    parser.add_argument("--play-batch-size", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Training steps per policy update.")
    parser.add_argument("--kl-targ", type=float, default=None)
    parser.add_argument("--check-freq", type=int, default=None)
    parser.add_argument("--tactic-ratio", type=float, default=None, help="Fraction of each training batch generated from hard-coded tactical positions. Default profile value is 0.20.")
    parser.add_argument("--no-tactic-data", action="store_true", help="Disable generated tactical samples in policy updates.")
    parser.add_argument("--pure-mcts-playouts", dest="pure_mcts_playout_num", type=int, default=None)
    parser.add_argument("--pure-inc-threshold", type=float, default=None)
    parser.add_argument("--pure-inc-step", type=int, default=None)
    parser.add_argument("--pure-max", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None, help="Network trunk channels for new models.")
    parser.add_argument("--n-blocks", type=int, default=None, help="Residual block count for new models.")
    parser.add_argument("--no-attention", action="store_true", help="Disable channel/spatial attention for new models.")
    parser.add_argument("--cpu", action="store_true", help="Allow CPU training instead of requiring CUDA.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA AMP mixed precision. Benchmark first; it is optional.")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last memory format. Benchmark first; it is optional.")
    parser.add_argument("--mcts-batch-size", type=int, default=None, help="Batch leaf evaluations inside MCTS.")
    parser.add_argument(
        "--playout-schedule",
        default="",
        help="Optional self-play playout schedule, e.g. 1:64,700:96,1400:140.",
    )
    parser.add_argument(
        "--difficulty-milestones",
        default="500,1000,2000",
        help="Comma-separated self-play batch checkpoints to save for PvE difficulty, e.g. 500,1000,2000 or none.",
    )
    parser.add_argument("--milestone-prefix", default="checker_policy")
    parser.add_argument(
        "--lite-net",
        action="store_true",
        help="Train a faster 64-channel, 5-block model without attention. Use a matching lite init model or train from scratch.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    resume_state = load_train_state(args.state_file) if args.continue_last else {}
    if args.continue_last:
        if not resume_state:
            raise RuntimeError("No training state found at {}".format(args.state_file))
        if args.init_model is None:
            current_model = resume_state.get("current_model_file")
            best_model = resume_state.get("best_model_file")
            if current_model and os.path.exists(current_model):
                args.init_model = current_model
            elif best_model and os.path.exists(best_model):
                args.init_model = best_model
        if args.playouts is None and resume_state.get("n_playout") is not None:
            args.playouts = int(resume_state["n_playout"])
        if args.mcts_batch_size is None and resume_state.get("mcts_batch_size") is not None:
            args.mcts_batch_size = int(resume_state["mcts_batch_size"])
        if args.max_moves is None and resume_state.get("max_moves") is not None:
            args.max_moves = int(resume_state["max_moves"])
        if args.buffer_file is None and resume_state.get("buffer_file"):
            args.buffer_file = resume_state["buffer_file"]
        if args.buffer_save_freq is None and resume_state.get("buffer_save_freq") is not None:
            args.buffer_save_freq = int(resume_state["buffer_save_freq"])
        resume_arg_specs = (
            ("learn_rate", "learn_rate", float),
            ("lr_multiplier", "lr_multiplier", float),
            ("temp", "temperature", float),
            ("c_puct", "c_puct", float),
            ("buffer_size", "buffer_size", int),
            ("batch_size", "batch_size", int),
            ("min_data_for_train", "min_data_for_train", int),
            ("train_freq", "train_freq", int),
            ("play_batch_size", "play_batch_size", int),
            ("eval_games", "eval_games", int),
            ("epochs", "epochs", int),
            ("kl_targ", "kl_targ", float),
            ("check_freq", "check_freq", int),
            ("tactic_ratio", "tactic_ratio", float),
            ("pure_mcts_playout_num", "pure_mcts_playout_num", int),
            ("pure_inc_threshold", "pure_inc_threshold", float),
            ("pure_inc_step", "pure_inc_step", int),
            ("pure_max", "pure_max", int),
        )
        for arg_name, state_key, caster in resume_arg_specs:
            if getattr(args, arg_name) is None and resume_state.get(state_key) is not None:
                setattr(args, arg_name, caster(resume_state[state_key]))
        if not args.playout_schedule and resume_state.get("playout_schedule"):
            args.playout_schedule = ",".join(
                "{}:{}".format(int(item[0]), int(item[1]))
                for item in resume_state["playout_schedule"]
            )
        print(
            "Continue last checker training: start_batch={}, init_model={}, n_playout={}, mcts_batch={}".format(
                int(resume_state.get("completed_batches", 0)),
                args.init_model,
                args.playouts,
                args.mcts_batch_size,
            )
        )

    milestone_checkpoints = build_milestone_checkpoints(args.difficulty_milestones, args.milestone_prefix)
    playout_schedule = build_playout_schedule(args.playout_schedule)
    overrides = {
        "current_model_file": args.current_model,
        "best_model_file": args.best_model,
        "learn_rate": args.learn_rate,
        "lr_multiplier": args.lr_multiplier,
        "temp": args.temp,
        "game_batch_num": args.game_batches,
        "n_playout": args.playouts,
        "c_puct": args.c_puct,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "min_data_for_train": args.min_data_for_train,
        "train_freq": args.train_freq,
        "play_batch_size": args.play_batch_size,
        "eval_games": args.eval_games,
        "max_moves": args.max_moves,
        "epochs": args.epochs,
        "kl_targ": args.kl_targ,
        "check_freq": args.check_freq,
        "tactic_ratio": 0.0 if args.no_tactic_data else args.tactic_ratio,
        "pure_mcts_playout_num": args.pure_mcts_playout_num,
        "pure_inc_threshold": args.pure_inc_threshold,
        "pure_inc_step": args.pure_inc_step,
        "pure_max": args.pure_max,
        "channels": args.channels,
        "n_blocks": args.n_blocks,
        "milestone_checkpoints": milestone_checkpoints,
        "playout_schedule": playout_schedule,
        "use_amp": args.amp,
        "channels_last": args.channels_last,
        "mcts_batch_size": args.mcts_batch_size,
        "state_file": args.state_file,
        "buffer_file": args.buffer_file,
        "buffer_save_freq": args.buffer_save_freq,
        "restore_buffer": args.continue_last and not args.no_restore_buffer,
    }
    if resume_state:
        overrides["start_batch"] = int(resume_state.get("completed_batches", 0))
        overrides["best_win_ratio"] = float(resume_state.get("best_win_ratio", 0.0))
    if args.lite_net:
        overrides["channels"] = 64
        overrides["n_blocks"] = 5
        overrides["use_attention"] = False
    if args.channels is not None:
        overrides["channels"] = args.channels
    if args.n_blocks is not None:
        overrides["n_blocks"] = args.n_blocks
    if args.no_attention:
        overrides["use_attention"] = False

    use_gpu = not args.cpu
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install CUDA PyTorch or rerun with --cpu for debugging.")

    TrainPipeline(init_model=args.init_model, profile=args.profile, use_gpu=use_gpu, **overrides).run()


if __name__ == "__main__":
    main()
