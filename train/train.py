# -*- coding: utf-8 -*-
"""AlphaZero-style training pipeline for Gomoku.

Default profile now trains standard 15x15 Gomoku. Use --profile classic for
longer, stronger runs.
"""

from __future__ import print_function

import argparse
import json
import os
import pickle
import random
import sys
from collections import defaultdict, deque

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from games.gomoku_game import Board, Game
from pure_mcts.mcts_pure import MCTSPlayer as MCTS_Pure
from _mcts_alphazero.mcts_alphaZero import MCTSPlayer
from policy_value_net.gomoku_policy_value_net_pytorch import PolicyValueNet


GOMOKU_PROFILES = {
    "day": {
        "board_width": 15,
        "board_height": 15,
        "n_in_row": 5,
        "learn_rate": 8e-4,
        "temp": 1.0,
        "n_playout": 96,
        "c_puct": 2.5,
        "buffer_size": 24000,
        "batch_size": 256,
        "play_batch_size": 1,
        "epochs": 2,
        "kl_targ": 0.02,
        "check_freq": 50,
        "game_batch_num": 1000,
        "pure_mcts_playout_num": 200,
        "eval_games": 4,
        "pure_inc_threshold": 0.90,
        "pure_inc_step": 200,
        "pure_max": 1500,
        "channels": 128,
        "n_blocks": 10,
        "use_attention": True,
        "tactic_ratio": 0.20,
    },
    "balanced": {
        "board_width": 15,
        "board_height": 15,
        "n_in_row": 5,
        "learn_rate": 6e-4,
        "temp": 1.0,
        "n_playout": 160,
        "c_puct": 2.5,
        "buffer_size": 32000,
        "batch_size": 384,
        "play_batch_size": 1,
        "epochs": 3,
        "kl_targ": 0.025,
        "check_freq": 80,
        "game_batch_num": 2000,
        "pure_mcts_playout_num": 300,
        "eval_games": 6,
        "pure_inc_threshold": 0.90,
        "pure_inc_step": 300,
        "pure_max": 2500,
        "channels": 128,
        "n_blocks": 10,
        "use_attention": True,
        "tactic_ratio": 0.20,
    },
    "classic": {
        "board_width": 15,
        "board_height": 15,
        "n_in_row": 5,
        "learn_rate": 5e-4,
        "temp": 1.0,
        "n_playout": 320,
        "c_puct": 2,
        "buffer_size": 40000,
        "batch_size": 512,
        "play_batch_size": 1,
        "epochs": 3,
        "kl_targ": 0.025,
        "check_freq": 100,
        "game_batch_num": 3500,
        "pure_mcts_playout_num": 800,
        "eval_games": 10,
        "pure_inc_threshold": 1.0,
        "pure_inc_step": 500,
        "pure_max": 5000,
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
        if profile not in GOMOKU_PROFILES:
            raise ValueError("unknown profile: {}".format(profile))

        cfg = dict(GOMOKU_PROFILES[profile])
        cfg.update({key: value for key, value in overrides.items() if value is not None})

        self.profile = profile
        self.board_width = int(cfg["board_width"])
        self.board_height = int(cfg["board_height"])
        self.n_in_row = int(cfg["n_in_row"])
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)

        self.learn_rate = float(cfg["learn_rate"])
        self.lr_multiplier = float(cfg.get("lr_multiplier", 1.0))
        self.temp = float(cfg["temp"])
        self.n_playout = int(cfg["n_playout"])
        self.c_puct = float(cfg["c_puct"])
        self.buffer_size = int(cfg["buffer_size"])
        self.batch_size = int(cfg["batch_size"])
        self.data_buffer = deque(maxlen=self.buffer_size)
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
        self.current_model_file = cfg.get("current_model_file", "./current_policy15.model")
        self.best_model_file = cfg.get("best_model_file", "./best_policy15.model")
        self.milestone_checkpoints = dict(cfg.get("milestone_checkpoints", {}))
        self.playout_schedule = list(cfg.get("playout_schedule", []))
        self.start_batch = int(cfg.get("start_batch", 0))
        self.state_file = cfg.get("state_file", "./gomoku15_train_state.json")
        self.use_gpu = bool(use_gpu)
        self.channels = int(cfg["channels"])
        self.n_blocks = int(cfg["n_blocks"])
        self.use_attention = bool(cfg["use_attention"])
        self.use_amp = bool(cfg.get("use_amp", False))
        self.channels_last = bool(cfg.get("channels_last", False))
        self.mcts_batch_size = int(cfg.get("mcts_batch_size", 8))
        self.tactic_ratio = max(0.0, min(0.5, float(cfg.get("tactic_ratio", 0.0))))
        self.buffer_file = cfg.get("buffer_file", "./gomoku15_data_buffer.pkl")
        self.buffer_save_freq = int(cfg.get("buffer_save_freq", 20))
        self.restore_buffer = bool(cfg.get("restore_buffer", False))

        self.policy_value_net = PolicyValueNet(
            self.board_width,
            self.board_height,
            model_file=init_model,
            use_gpu=self.use_gpu,
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
            "n_in_row": self.n_in_row,
            "learn_rate": float(self.learn_rate),
            "lr_multiplier": float(self.lr_multiplier),
            "temperature": float(self.temp),
            "c_puct": float(self.c_puct),
            "n_playout": int(self.n_playout),
            "mcts_batch_size": int(self.mcts_batch_size),
            "playout_schedule": list(self.playout_schedule),
            "buffer_size": int(self.buffer_size),
            "batch_size": int(self.batch_size),
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
            "game": "gomoku",
            "board_width": int(self.board_width),
            "board_height": int(self.board_height),
            "n_in_row": int(self.n_in_row),
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
        if not isinstance(payload, dict) or payload.get("game") != "gomoku":
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

    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(
                    np.flipud(mcts_prob.reshape(self.board_height, self.board_width)),
                    i,
                )
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))

                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def _line_moves(self, start_r, start_c, dr, dc, length):
        return [
            (start_r + i * dr) * self.board_width + (start_c + i * dc)
            for i in range(length)
        ]

    def _random_five_segment(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for _ in range(100):
            dr, dc = random.choice(directions)
            min_r = 0
            max_r = self.board_height - 1 - dr * (self.n_in_row - 1)
            min_c = 0
            max_c = self.board_width - 1 - dc * (self.n_in_row - 1)
            if dc < 0:
                min_c = self.n_in_row - 1
                max_c = self.board_width - 1
            if max_r < min_r or max_c < min_c:
                continue
            r = random.randint(min_r, max_r)
            c = random.randint(min_c, max_c)
            return self._line_moves(r, c, dr, dc, self.n_in_row)
        raise RuntimeError("failed to build gomoku tactic segment")

    def _state_from_moves(self, current_moves, opponent_moves, last_move=-1, current_is_p1=True):
        state = np.zeros((4, self.board_width, self.board_height), dtype=np.float32)
        for move in current_moves:
            r = move // self.board_width
            c = move % self.board_width
            state[0, self.board_height - 1 - r, c] = 1.0
        for move in opponent_moves:
            r = move // self.board_width
            c = move % self.board_width
            state[1, self.board_height - 1 - r, c] = 1.0
        if last_move >= 0:
            r = last_move // self.board_width
            c = last_move % self.board_width
            state[2, self.board_height - 1 - r, c] = 1.0
        if current_is_p1:
            state[3, :, :] = 1.0
        return state

    def _add_small_random_fill(self, current_moves, opponent_moves, reserved):
        occupied = set(current_moves) | set(opponent_moves) | set(reserved)
        empties = [m for m in range(self.board_width * self.board_height) if m not in occupied]
        random.shuffle(empties)
        fill_pairs = random.randint(0, 3)
        for i in range(fill_pairs * 2):
            if not empties:
                break
            move = empties.pop()
            if i % 2 == 0:
                current_moves.add(move)
            else:
                opponent_moves.add(move)

    def _make_gomoku_tactic_sample(self):
        segment = self._random_five_segment()
        target = random.choice(segment)
        current_moves = set()
        opponent_moves = set()
        if random.random() < 0.55:
            # Immediate win: four current stones and one finishing point.
            current_moves.update(m for m in segment if m != target)
            value = 1.0
        else:
            # Emergency block: opponent has four in a five-line segment.
            opponent_moves.update(m for m in segment if m != target)
            value = 0.0

        self._add_small_random_fill(current_moves, opponent_moves, set(segment))
        last_move = random.choice(list((current_moves | opponent_moves) - {target})) if (current_moves or opponent_moves) else -1
        probs = np.zeros(self.board_width * self.board_height, dtype=np.float32)
        probs[target] = 1.0
        state = self._state_from_moves(
            current_moves,
            opponent_moves,
            last_move=last_move,
            current_is_p1=(random.random() < 0.5),
        )
        return state, probs, value

    def _make_gomoku_open_three_sample(self, defense=None):
        segment = self._random_five_segment()
        middle = segment[1:4]
        targets = [segment[0], segment[4]]
        current_moves = set()
        opponent_moves = set()
        if defense is None:
            defense = random.random() >= 0.55
        if not defense:
            current_moves.update(middle)
            value = 0.45
        else:
            opponent_moves.update(middle)
            value = -0.15

        self._add_small_random_fill(current_moves, opponent_moves, set(segment))
        occupied = list((current_moves | opponent_moves) - set(targets))
        last_move = random.choice(occupied) if occupied else -1
        probs = np.zeros(self.board_width * self.board_height, dtype=np.float32)
        for target in targets:
            probs[target] = 0.5
        state = self._state_from_moves(
            current_moves,
            opponent_moves,
            last_move=last_move,
            current_is_p1=(random.random() < 0.5),
        )
        return state, probs, value

    def _random_center_for_double_threat(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for _ in range(100):
            r = random.randint(2, self.board_height - 3)
            c = random.randint(2, self.board_width - 3)
            dirs = random.sample(directions, 2)
            ok = True
            support = []
            extensions = []
            for dr, dc in dirs:
                for sign in (-1, 1):
                    sr, sc = r + sign * dr, c + sign * dc
                    er, ec = r + sign * 2 * dr, c + sign * 2 * dc
                    if not (0 <= sr < self.board_height and 0 <= sc < self.board_width):
                        ok = False
                    if not (0 <= er < self.board_height and 0 <= ec < self.board_width):
                        ok = False
                    support.append(sr * self.board_width + sc)
                    extensions.append(er * self.board_width + ec)
            if ok:
                return r * self.board_width + c, support, extensions
        return None, [], []

    def _make_gomoku_double_threat_sample(self):
        target, support, extensions = self._random_center_for_double_threat()
        if target is None:
            return self._make_gomoku_open_three_sample()

        current_moves = set()
        opponent_moves = set()
        defending = random.random() < 0.45
        if defending:
            opponent_moves.update(support)
            value = 0.05
        else:
            current_moves.update(support)
            value = 0.50

        reserved = set(support) | set(extensions) | {target}
        self._add_small_random_fill(current_moves, opponent_moves, reserved)
        occupied = list(current_moves | opponent_moves)
        last_move = random.choice(occupied) if occupied else -1

        probs = np.zeros(self.board_width * self.board_height, dtype=np.float32)
        probs[target] = 0.65
        side_weight = 0.35 / max(1, len(extensions))
        for move in extensions:
            probs[move] += side_weight
        probs /= np.sum(probs)

        state = self._state_from_moves(
            current_moves,
            opponent_moves,
            last_move=last_move,
            current_is_p1=(random.random() < 0.5),
        )
        return state, probs.astype(np.float32), value

    def _line_window(self, start_r, start_c, dr, dc, length):
        cells = []
        for i in range(length):
            r = start_r + i * dr
            c = start_c + i * dc
            if not (0 <= r < self.board_height and 0 <= c < self.board_width):
                return None
            cells.append(r * self.board_width + c)
        return cells

    def _count_open_threes_at(self, stones, blockers, anchor):
        stones = set(stones)
        blockers = set(blockers)
        if anchor not in stones:
            return 0
        ar = anchor // self.board_width
        ac = anchor % self.board_width
        count = 0
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            found_this_dir = False
            for offset in range(-4, 1):
                start_r = ar + offset * dr
                start_c = ac + offset * dc
                cells = self._line_window(start_r, start_c, dr, dc, self.n_in_row)
                if not cells or anchor not in cells:
                    continue
                if cells[0] in blockers or cells[-1] in blockers:
                    continue
                if cells[0] in stones or cells[-1] in stones:
                    continue
                middle = cells[1:-1]
                if all(cell not in blockers for cell in middle) and sum(cell in stones for cell in middle) == 3:
                    found_this_dir = True
                    break
            if found_this_dir:
                count += 1
        return count

    def _creates_double_open_three(self, stones, blockers, move):
        if move in stones or move in blockers:
            return False
        next_stones = set(stones)
        next_stones.add(move)
        return self._count_open_threes_at(next_stones, blockers, move) >= 2

    def _move_allows_opponent_fork(self, candidate, current_moves, opponent_moves, fork_move):
        if candidate in current_moves or candidate in opponent_moves:
            return False
        next_current = set(current_moves)
        next_current.add(candidate)
        if fork_move in next_current or fork_move in opponent_moves:
            return False
        return self._creates_double_open_three(opponent_moves, next_current, fork_move)

    def _make_gomoku_threat_avoidance_sample(self):
        for _ in range(100):
            target, support, extensions = self._random_center_for_double_threat()
            if target is None:
                break
            current_moves = set()
            opponent_moves = set(support)
            reserved = set(support) | set(extensions) | {target}
            self._add_small_random_fill(current_moves, opponent_moves, reserved)

            legal = [
                m for m in range(self.board_width * self.board_height)
                if m not in current_moves and m not in opponent_moves
            ]
            dangerous = {
                m for m in legal
                if self._move_allows_opponent_fork(m, current_moves, opponent_moves, target)
            }
            safe = [m for m in legal if m not in dangerous]
            critical = [m for m in [target] + list(extensions) if m in safe]
            if not dangerous or target not in critical:
                continue

            probs = np.zeros(self.board_width * self.board_height, dtype=np.float32)
            probs[target] = 0.70
            extension_safe = [m for m in extensions if m in safe and m != target]
            if extension_safe:
                for move in extension_safe:
                    probs[move] += 0.20 / len(extension_safe)
            other_safe = [m for m in safe if m not in critical]
            if other_safe:
                # Keep a small amount of mass for genuinely safe alternatives,
                # but never reward moves that allow the opponent fork.
                sample_safe = random.sample(other_safe, min(8, len(other_safe)))
                for move in sample_safe:
                    probs[move] += 0.10 / len(sample_safe)
            probs /= np.sum(probs)

            occupied = list(current_moves | opponent_moves)
            last_move = random.choice(occupied) if occupied else -1
            state = self._state_from_moves(
                current_moves,
                opponent_moves,
                last_move=last_move,
                current_is_p1=(random.random() < 0.5),
            )
            return state, probs.astype(np.float32), -0.25
        return self._make_gomoku_open_three_sample(defense=True)

    def _make_gomoku_soft_tactic_sample(self):
        if random.random() < 0.65:
            return self._make_gomoku_open_three_sample()
        return self._make_gomoku_double_threat_sample()

    def _generate_tactic_batch(self, count):
        samples = []
        for _ in range(max(0, int(count))):
            roll = random.random()
            if roll < 0.25:
                samples.append(self._make_gomoku_tactic_sample())
            elif roll < 0.55:
                samples.append(self._make_gomoku_open_three_sample(defense=True))
            elif roll < 0.70:
                samples.append(self._make_gomoku_open_three_sample(defense=False))
            elif roll < 0.90:
                samples.append(self._make_gomoku_threat_avoidance_sample())
            else:
                samples.append(self._make_gomoku_double_threat_sample())
        return samples

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
            (
                "kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},"
                "explained_var_old:{:.3f},explained_var_new:{:.3f}"
            ).format(kl, self.lr_multiplier, loss, entropy, 1 - old_var, 1 - new_var)
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
            "num_playouts:{}, win:{}, lose:{}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1],
                win_cnt[2],
                win_cnt[-1],
            )
        )
        return win_ratio

    def run(self):
        print(
                "profile:{}, board:{}x{}, start_batch:{}, n_playout:{}, mcts_batch:{}, games:{}, batch:{}, tactic_ratio:{:.2f}, update_epochs:{}, eval_games:{}, net:{}x{}, attention:{}".format(
                self.profile,
                self.board_width,
                self.board_height,
                self.start_batch,
                self.n_playout,
                self.mcts_batch_size,
                self.game_batch_num,
                self.batch_size,
                self.tactic_ratio,
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
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()

                if batch_i in self.milestone_checkpoints:
                    checkpoint_file = self.milestone_checkpoints[batch_i]
                    self.policy_value_net.save_model(checkpoint_file)
                    print("Save milestone checkpoint: batch {} -> {}".format(batch_i, checkpoint_file))

                if batch_i % self.check_freq == 0:
                    print("current self-play batch: {}".format(batch_i))
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
                        self.best_win_ratio = 0.0
                        print("Increase pure_mcts_playout_num ->", self.pure_mcts_playout_num)
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
    parser = argparse.ArgumentParser(description="Train Gomoku policy-value network.")
    parser.add_argument("--profile", choices=sorted(GOMOKU_PROFILES), default="day")
    parser.add_argument("--init-model", default=None, help="Resume from an existing .model file.")
    parser.add_argument("--continue-last", action="store_true", help="Continue from the state recorded in --state-file.")
    parser.add_argument("--state-file", default="./gomoku15_train_state.json", help="JSON file used to record resume metadata.")
    parser.add_argument("--buffer-file", default=None, help="Pickle file used to save and restore the self-play replay buffer.")
    parser.add_argument("--buffer-save-freq", type=int, default=None, help="Save the replay buffer every N completed self-play batches. Use 0 to only save on Ctrl+C/end.")
    parser.add_argument("--no-restore-buffer", action="store_true", help="With --continue-last, skip loading the replay buffer file.")
    parser.add_argument("--current-model", default="./current_policy15.model")
    parser.add_argument("--best-model", default="./best_policy15.model")
    parser.add_argument("--board-size", type=int, default=None, help="Set width and height together.")
    parser.add_argument("--n-in-row", type=int, default=None)
    parser.add_argument("--learn-rate", type=float, default=None)
    parser.add_argument("--lr-multiplier", type=float, default=None)
    parser.add_argument("--temperature", "--temp", dest="temp", type=float, default=None)
    parser.add_argument("--game-batches", type=int, default=None)
    parser.add_argument("--playouts", type=int, default=None)
    parser.add_argument("--c-puct", type=float, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--play-batch-size", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
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
        help="Optional self-play playout schedule, e.g. 1:80,700:120,1400:180.",
    )
    parser.add_argument(
        "--difficulty-milestones",
        default="500,1000,2000",
        help="Comma-separated self-play batch checkpoints to save for PvE difficulty, e.g. 500,1000,2000 or none.",
    )
    parser.add_argument("--milestone-prefix", default="gomoku15_policy")
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
            "Continue last training: start_batch={}, init_model={}, n_playout={}, mcts_batch={}".format(
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
        "n_in_row": args.n_in_row,
        "learn_rate": args.learn_rate,
        "lr_multiplier": args.lr_multiplier,
        "temp": args.temp,
        "game_batch_num": args.game_batches,
        "n_playout": args.playouts,
        "c_puct": args.c_puct,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "play_batch_size": args.play_batch_size,
        "eval_games": args.eval_games,
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
    if args.board_size is not None:
        overrides["board_width"] = args.board_size
        overrides["board_height"] = args.board_size
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
