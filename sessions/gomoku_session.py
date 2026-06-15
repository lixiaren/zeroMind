# -*- coding: utf-8 -*-
import random
from sessions.base_session import BaseSession

# 按你的实际路径调整
from games.gomoku_game import Board  # 或：from games.game import Board


class GomokuSession(BaseSession):
    def __init__(self, mode: str, size=15, n_in_row=5, p1=None, p2=None, human_side=1, ai_agent=None):
        super().__init__("gomoku", mode)
        self.size = int(size)
        self.n_in_row = int(n_in_row)

        # 用Board 作为权威规则引擎
        self.board = Board(width=self.size, height=self.size, n_in_row=self.n_in_row)
        self.board.init_board(start_player=0)  # 永远让 player=1 先手
        self.start_player = 0
        self.move_history = []  # list of engine indices
        # players（pvp: p1/p2 为用户名；pve: p1=用户名, p2="AI"）
        self.p1 = p1
        self.p2 = p2
        self.human_side = int(human_side)  # pve: human is 1 or 2
        self.ai_agent = ai_agent           

    # --------- helpers ----------
    def _board_list(self):
        """把 Board.states 转成 list[int]，仅用于兼容老 ai_agent 接口。"""
        arr = [0] * (self.size * self.size)
        for idx, player in self.board.states.items():
            arr[int(idx)] = int(player)
        return arr

    @property
    def current_player(self):
        return int(self.board.current_player)

    @property
    def last_move(self):
        return int(self.board.last_move)

    # --------- protocol ----------
    def state_payload(self):
        states = [{"index": int(i), "player": int(v)} for i, v in self.board.states.items()]
        return {
            "game": "gomoku",
            "matchId": self.match_id,
            "states": states,
            "currentPlayer": self.current_player,
            "lastMove": self.last_move,
        }

    def make_move(self, idx: int, player: int):
        if self.ended:
            return {"ok": False, "msg": "The game is already over."}

        idx = int(idx)
        player = int(player)

        if player != self.current_player:
            return {"ok": False, "msg": "It is not your turn."}

        if idx not in self.board.availables:
            return {"ok": False, "msg": "Invalid move. Please choose an empty intersection."}

        # 落子
        self.board.do_move(idx)

        
        self.move_history.append(idx)   # 关键：不记历史就没法撤销

        # 通知 AI：对手刚走了 idx（树跟上）
        if self.mode == "pve" and self.ai_agent and hasattr(self.ai_agent, "update_with_move"):
            self.ai_agent.update_with_move(idx)

        end, winner = self.board.game_end()
        if end:
            self.ended = True
            self.winner = int(winner)
            return {"ok": True, "winner": self.winner}

        return {"ok": True, "winner": 0}

    def ai_move_if_needed(self):
        if self.mode != "pve" or self.ended:
            return None

        ai_side = 2 if self.human_side == 1 else 1
        if self.current_player != ai_side:
            return None

        if not self.board.availables:
            self.ended = True
            self.winner = -1
            return {"index": -1, "player": ai_side, "winner": -1}

        if self.ai_agent and hasattr(self.ai_agent, "get_action"):
            idx = int(self.ai_agent.get_action(self.board))
            if idx not in self.board.availables:
                # AI 出非法步就拒绝并让它重新选（这里简单兜底成第一个合法步）
                idx = int(self.board.availables[0])
        else:
            # 兜底：没注入模型就随机
            import random
            idx = int(random.choice(self.board.availables))

        res = self.make_move(idx, ai_side)
        return {"index": idx, "player": ai_side, "winner": int(res.get("winner", 0))}
    
    
    def undo(self, steps: int = 1):
        """撤销最近 steps 手（Gomoku 一步 = 一个 index）。"""
        steps = int(steps)
        if steps <= 0:
            return {"ok": True}

        if len(self.move_history) < steps:
            return {"ok": False, "msg": "There are no moves to undo."}

        # 1) 弹出历史
        for _ in range(steps):
            self.move_history.pop()

        # 2) 重建 Board（最稳，不依赖 Board 的 undo）
        self.board = Board(width=self.size, height=self.size, n_in_row=self.n_in_row)
        self.board.init_board(start_player=self.start_player)
        for mv in self.move_history:
            self.board.do_move(int(mv))

        # 3) 对局状态回到进行中
        self.ended = False
        self.winner = 0

        # 4) AI 树同步（能 reset 就 reset；否则把历史喂回去）
        if self.mode == "pve" and self.ai_agent:
            if hasattr(self.ai_agent, "reset"):
                try:
                    self.ai_agent.reset()
                except Exception:
                    pass
            if hasattr(self.ai_agent, "update_with_move"):
                try:
                    for mv in self.move_history:
                        self.ai_agent.update_with_move(int(mv))
                except Exception:
                    pass

        return {"ok": True}
