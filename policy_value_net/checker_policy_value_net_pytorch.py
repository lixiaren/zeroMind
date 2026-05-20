# -*- coding: utf-8 -*-
"""
PyTorch policy-value network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def set_learning_rate(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_attention: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.use_attention = bool(use_attention)
        if self.use_attention:
            self.ca = ChannelAttention(channels)
            self.sa = SpatialAttention(kernel_size=7)
        self.skip_attention = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.use_attention and not self.skip_attention:
            out = self.ca(out)
            out = self.sa(out)
        out = out + identity
        return F.relu(out, inplace=True)


class Net(nn.Module):
    def __init__(
        self,
        board_width: int,
        board_height: int,
        action_size: int,
        in_channels: int = 4,
        channels: int = 128,
        n_blocks: int = 10,
        use_attention: bool = True,
        skip_attention: bool = False,
    ):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.action_size = action_size
        self.channels = int(channels)
        self.n_blocks = int(n_blocks)
        self.use_attention = bool(use_attention)

        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, use_attention=self.use_attention) for _ in range(n_blocks)]
        )

        # policy head
        self.p_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * board_width * board_height, action_size)

        # value head
        self.v_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_fc1 = nn.Linear(32 * board_width * board_height, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, state_input: torch.Tensor):
        x = F.relu(self.bn_in(self.conv_in(state_input)), inplace=True)
        x = self.res_blocks(x)

        p = F.relu(self.p_bn(self.p_conv(x)), inplace=True)
        p = p.reshape(p.size(0), -1)
        log_act_probs = F.log_softmax(self.p_fc(p), dim=1)

        v = F.relu(self.v_bn(self.v_conv(x)), inplace=True)
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.v_fc1(v), inplace=True)
        value = torch.tanh(self.v_fc2(v))
        return log_act_probs, value


def _infer_arch_from_state_dict(net_params):
    conv_in = net_params.get("conv_in.weight")
    if conv_in is None:
        return None

    channels = int(conv_in.shape[0])
    in_channels = int(conv_in.shape[1])
    block_ids = set()
    for key in net_params.keys():
        if key.startswith("res_blocks."):
            parts = key.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                block_ids.add(int(parts[1]))
    n_blocks = max(block_ids) + 1 if block_ids else 0
    use_attention = any(".ca." in key or ".sa." in key for key in net_params.keys())
    return channels, n_blocks, in_channels, use_attention


class PolicyValueNet:
    def __init__(
        self,
        board_width,
        board_height,
        model_file=None,
        use_gpu=False,
        action_size=None,
        in_channels: int = 4,
        channels: int = 128,
        n_blocks: int = 10,
        use_attention: bool = True,
        skip_attention: bool = False,
        use_amp: bool = False,
        channels_last: bool = False,
    ):
        self.use_gpu = bool(use_gpu) and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.board_width = int(board_width)
        self.board_height = int(board_height)
        self.in_channels = int(in_channels)
        self.action_size = int(action_size) if action_size is not None else self.board_width * self.board_height
        self.l2_const = 1e-4
        self.use_amp = bool(use_amp) and self.use_gpu
        self.channels_last = bool(channels_last) and self.use_gpu
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        net_params = None
        if model_file:
            net_params = torch.load(model_file, map_location=self.device)
            inferred = _infer_arch_from_state_dict(net_params)
            if inferred is not None:
                channels, n_blocks, inferred_in_channels, use_attention = inferred
                self.in_channels = inferred_in_channels

        self.policy_value_net = Net(
            self.board_width,
            self.board_height,
            action_size=self.action_size,
            in_channels=self.in_channels,
            channels=channels,
            n_blocks=n_blocks,
            use_attention=use_attention,
        ).to(self.device)
        if self.channels_last:
            self.policy_value_net = self.policy_value_net.to(memory_format=torch.channels_last)
        self.set_skip_attention(skip_attention)

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if net_params is not None:
            self.policy_value_net.load_state_dict(net_params)

    def set_skip_attention(self, skip_attention: bool):
        for module in self.policy_value_net.modules():
            if isinstance(module, ResidualBlock):
                module.skip_attention = bool(skip_attention)

    def policy_value(self, state_batch):
        state_batch = np.asarray(state_batch, dtype=np.float32)
        state_tensor = torch.from_numpy(state_batch).to(self.device)
        if self.channels_last:
            state_tensor = state_tensor.contiguous(memory_format=torch.channels_last)
        was_training = self.policy_value_net.training
        self.policy_value_net.eval()
        with torch.inference_mode():
            with autocast("cuda", enabled=self.use_amp):
                log_act_probs, value = self.policy_value_net(state_tensor)
            act_probs = torch.exp(log_act_probs).cpu().numpy()
            value_np = value.cpu().numpy()
        if was_training:
            self.policy_value_net.train()
        return act_probs, value_np

    def policy_value_fn(self, board):
        legal_actions = list(board.availables)
        current_state = np.ascontiguousarray(
            board.current_state().reshape(-1, self.in_channels, self.board_width, self.board_height),
            dtype=np.float32,
        )
        state_tensor = torch.from_numpy(current_state).to(self.device)
        if self.channels_last:
            state_tensor = state_tensor.contiguous(memory_format=torch.channels_last)
        was_training = self.policy_value_net.training
        self.policy_value_net.eval()
        with torch.inference_mode():
            with autocast("cuda", enabled=self.use_amp):
                log_act_probs, value = self.policy_value_net(state_tensor)
            act_probs = torch.exp(log_act_probs).cpu().numpy().flatten()
            value = float(value.cpu().numpy()[0][0])
        if was_training:
            self.policy_value_net.train()

        if not legal_actions:
            return [], value

        legal_probs = act_probs[legal_actions]
        prob_sum = float(np.sum(legal_probs))
        if prob_sum > 1e-12:
            legal_probs = legal_probs / prob_sum
        else:
            legal_probs = np.ones(len(legal_actions), dtype=np.float32) / len(legal_actions)
        return zip(legal_actions, legal_probs), value

    def policy_value_fn_batch(self, boards):
        boards = list(boards)
        if not boards:
            return []

        state_batch = np.asarray(
            [
                board.current_state().reshape(
                    self.in_channels,
                    self.board_width,
                    self.board_height,
                )
                for board in boards
            ],
            dtype=np.float32,
        )
        state_tensor = torch.from_numpy(state_batch).to(self.device)
        if self.channels_last:
            state_tensor = state_tensor.contiguous(memory_format=torch.channels_last)

        was_training = self.policy_value_net.training
        self.policy_value_net.eval()
        with torch.inference_mode():
            with autocast("cuda", enabled=self.use_amp):
                log_act_probs, value = self.policy_value_net(state_tensor)
            act_probs_batch = torch.exp(log_act_probs).cpu().numpy()
            value_batch = value.cpu().numpy().reshape(-1)
        if was_training:
            self.policy_value_net.train()

        results = []
        for board, act_probs, value in zip(boards, act_probs_batch, value_batch):
            legal_actions = list(board.availables)
            if not legal_actions:
                results.append(([], float(value)))
                continue

            legal_probs = act_probs[legal_actions]
            prob_sum = float(np.sum(legal_probs))
            if prob_sum > 1e-12:
                legal_probs = legal_probs / prob_sum
            else:
                legal_probs = np.ones(len(legal_actions), dtype=np.float32) / len(legal_actions)
            results.append((list(zip(legal_actions, legal_probs)), float(value)))
        return results

    def train_step(self, state_batch, mcts_probs, winner_batch, lr: float):
        state_batch = torch.as_tensor(np.asarray(state_batch, dtype=np.float32), device=self.device)
        mcts_probs = torch.as_tensor(np.asarray(mcts_probs, dtype=np.float32), device=self.device)
        winner_batch = torch.as_tensor(np.asarray(winner_batch, dtype=np.float32), device=self.device)
        if self.channels_last:
            state_batch = state_batch.contiguous(memory_format=torch.channels_last)

        self.optimizer.zero_grad(set_to_none=True)
        set_learning_rate(self.optimizer, lr)

        self.policy_value_net.train()
        with autocast("cuda", enabled=self.use_amp):
            log_act_probs, value = self.policy_value_net(state_batch)
            value_loss = F.mse_loss(value.view(-1), winner_batch)
            policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
            loss = value_loss + policy_loss

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1))
        return loss.item(), entropy.item()

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def save_model(self, model_file):
        torch.save(self.get_policy_param(), model_file)
