# -*- coding: utf-8 -*-
import random
from sessions.base_session import BaseSession
from games.checker_game import Board, EMPTY, P1_MAN, P2_MAN, P1_KING, P2_KING


class CheckersSession(BaseSession):
    """
    后端权威规则引擎：复用训练用的 checker_game.py

    前端发：
      make_move: {matchId, game:'checkers', from:0..63, to:0..63}

    后端 sync_state 给前端：
      board: [64] int
      currentPlayer: 1/2
      chainPos: int|null

      + 新增（渲染用）：
      lastFrom, lastTo: int|null
      legalMoves: Array<[from,to]>
      legalFrom: int[]
      legalToMap: { "from": [to...] }
      mustCapture: bool
      inChain: bool
    """

    def __init__(
        self,
        mode: str,
        p1=None,
        p2=None,
        human_player: int = 1,
        ai_agent=None,
        max_moves: int = 200,
        promote_ends_turn: bool = True,
    ):
        super().__init__("checkers", mode)

        self.p1 = p1
        self.p2 = p2
        self.human_player = int(human_player)
        self.ai_agent = ai_agent

        self.board = Board(width=8, height=8, max_moves=max_moves, promote_ends_turn=promote_ends_turn)
        self.board.init_board(start_player=0)  # 永远 P1 先手

    # ---------- helpers ----------
    @staticmethod
    def _engine_to_rc(engine_idx: int):
        r = int(engine_idx) // 8
        c = int(engine_idx) % 8
        return r, c

    def _engine_to_dark(self, engine_idx: int) -> int:
        r, c = self._engine_to_rc(engine_idx)
        return int(self.board._rc_to_dark[r, c])  # 0..31 or -1

    def _move_engine_to_action(self, frm_engine: int, to_engine: int) -> int:
        fd = self._engine_to_dark(frm_engine)
        td = self._engine_to_dark(to_engine)
        if fd < 0 or td < 0:
            return -1
        return fd * 32 + td  # 0..1023

    def _chain_pos_engine(self):
        if getattr(self.board, "_chain_pos", None) is None:
            return None
        r, c = self.board._chain_pos
        return int(r) * 8 + int(c)

    def _board_list64(self):
        # checker_game.Board.board: shape (8,8), row0 is bottom
        return [int(x) for x in self.board.board.reshape(-1).tolist()]

    def _action_to_engine_pair(self, action: int):
        fd, td = int(action) // 32, int(action) % 32
        fr, fc = self.board._dark_to_rc[fd]
        tr, tc = self.board._dark_to_rc[td]
        frm = int(fr) * 8 + int(fc)
        to = int(tr) * 8 + int(tc)
        is_cap = (abs(int(tr) - int(fr)) == 2 and abs(int(tc) - int(fc)) == 2)
        return frm, to, is_cap

    @property
    def current_player(self) -> int:
        return int(self.board.current_player)

    def _invalid_move_message(self, action: int) -> str:
        legal_actions = list(self.board.availables or [])
        if getattr(self.board, "_chain_pos", None) is not None:
            return "You must continue the capture chain with the highlighted piece."

        legal_from = {int(a) // 32 for a in legal_actions}
        capture_required = any(self._action_to_engine_pair(a)[2] for a in legal_actions)
        if capture_required:
            return "Capture is mandatory. Select a highlighted piece and capture target."

        from_dark = int(action) // 32 if action >= 0 else -1
        if from_dark not in legal_from:
            return "This piece has no legal move right now. Select a highlighted piece."

        return "Invalid target. Select a highlighted destination square."

    # ---------- payload ----------
    def state_payload(self):
        legal_actions = list(self.board.availables or [])

        legal_moves = []
        legal_from_set = set()
        legal_to_map = {}
        must_capture = False

        for a in legal_actions:
            frm, to, is_cap = self._action_to_engine_pair(a)
            legal_moves.append([frm, to])
            legal_from_set.add(frm)
            legal_to_map.setdefault(str(frm), []).append(to)
            if is_cap:
                must_capture = True

        in_chain = self.board._chain_pos is not None
        if in_chain:
            must_capture = True

        last_from = None
        last_to = None
        if int(getattr(self.board, "last_move", -1)) >= 0:
            lf, lt, _ = self._action_to_engine_pair(int(self.board.last_move))
            last_from, last_to = lf, lt

        return {
            "game": "checkers",
            "matchId": self.match_id,
            "board": self._board_list64(),
            "currentPlayer": self.current_player,
            "chainPos": self._chain_pos_engine(),
            "moveCount": int(getattr(self.board, "move_count", 0)),

            # ===== 新增：渲染提示 =====
            "lastFrom": last_from,
            "lastTo": last_to,
            "legalMoves": legal_moves,
            "legalFrom": sorted(list(legal_from_set)),
            "legalToMap": legal_to_map,
            "mustCapture": bool(must_capture),
            "inChain": bool(in_chain),
        }

    # ---------- gameplay ----------
    def make_move(self, frm: int, to: int, player: int):
        if self.ended:
            return {"ok": False, "msg": "The game is already over."}

        player = int(player)
        if player != self.current_player:
            return {"ok": False, "msg": "It is not your turn."}

        action = self._move_engine_to_action(frm, to)
        if action < 0:
            return {"ok": False, "msg": "Invalid square. Checkers pieces can only move on dark squares."}

        if action not in (self.board.availables or []):
            return {"ok": False, "msg": self._invalid_move_message(action)}

        self.board.do_move(int(action))

        end, winner = self.board.game_end()
        if end:
            self.ended = True
            self.winner = int(winner)
            return {"ok": True, "winner": self.winner}

        return {"ok": True, "winner": 0}

    def ai_move_if_needed(self):
        if self.mode != "pve" or self.ended:
            return None

        ai_player = 2 if self.human_player == 1 else 1
        if self.current_player != ai_player:
            return None

        legal = list(self.board.availables or [])
        if not legal:
            self.ended = True
            self.winner = 1 if ai_player == 2 else 2
            return {"action": -1, "player": ai_player, "winner": self.winner}

        # 真 AI 优先，否则随机兜底
        if self.ai_agent and hasattr(self.ai_agent, "get_action"):
            try:
                action = int(self.ai_agent.get_action(self.board))
            except Exception:
                action = int(random.choice(legal))
            if action not in legal:
                action = int(random.choice(legal))
        else:
            action = int(random.choice(legal))

        self.board.do_move(action)

        end, winner = self.board.game_end()
        if end:
            self.ended = True
            self.winner = int(winner)
            return {"action": action, "player": ai_player, "winner": self.winner}

        return {"action": action, "player": ai_player, "winner": 0}
