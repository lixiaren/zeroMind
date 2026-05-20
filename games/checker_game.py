# -*- coding: utf-8 -*-
"""
Checkers.
"""

from __future__ import print_function
import numpy as np

EMPTY = 0
P1_MAN = 1
P2_MAN = 2
P1_KING = 3
P2_KING = 4


def _is_p1(piece: int) -> bool:
    return piece in (P1_MAN, P1_KING)


def _is_p2(piece: int) -> bool:
    return piece in (P2_MAN, P2_KING)


def _is_king(piece: int) -> bool:
    return piece in (P1_KING, P2_KING)


class Board(object):
    def __init__(self, **kwargs):
        self.width = int(kwargs.get("width", 8))
        self.height = int(kwargs.get("height", 8))
        if self.width != 8 or self.height != 8:
            raise ValueError("Checkers board must be 8x8.")

        self.players = [1, 2]
        self.action_size = 32 * 32

        self.max_moves = int(kwargs.get("max_moves", 200))
        self.promote_ends_turn = bool(kwargs.get("promote_ends_turn", True))

        self.board = np.zeros((8, 8), dtype=np.int8)  # row0 is bottom
        self.current_player = 1
        self.last_move = -1
        self.move_count = 0

        self._chain_pos = None  # (r,c) if multi-capture must continue

        # compatibility fields
        self.states = {}
        self.availables = []

        # dark-square mappings
        self._dark_to_rc = []
        self._rc_to_dark = -np.ones((8, 8), dtype=np.int16)
        idx = 0
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 1:  # dark
                    self._dark_to_rc.append((r, c))
                    self._rc_to_dark[r, c] = idx
                    idx += 1
        assert idx == 32

    def __deepcopy__(self, memo):
        copied = type(self)(
            width=self.width,
            height=self.height,
            max_moves=self.max_moves,
            promote_ends_turn=self.promote_ends_turn,
        )
        copied.players = self.players[:]
        copied.action_size = self.action_size
        copied.board = self.board.copy()
        copied.current_player = self.current_player
        copied.last_move = self.last_move
        copied.move_count = self.move_count
        copied._chain_pos = None if self._chain_pos is None else tuple(self._chain_pos)
        copied.states = self.states.copy()
        copied.availables = list(self.availables or [])
        copied._dark_to_rc = self._dark_to_rc
        copied._rc_to_dark = self._rc_to_dark
        return copied

    def move_to_location(self, move: int):
        # returns [from_dark, to_dark]
        if move < 0 or move >= self.action_size:
            return [-1, -1]
        return [move // 32, move % 32]

    def location_to_move(self, location):
        # interpret input as [from_dark, to_dark]
        if len(location) != 2:
            return -1
        fr, to = int(location[0]), int(location[1])
        if fr < 0 or fr >= 32 or to < 0 or to >= 32:
            return -1
        return fr * 32 + to

    def init_board(self, start_player=0):
        if start_player not in (0, 1):
            raise ValueError("start_player must be 0 or 1")
        self.current_player = self.players[start_player]
        self.last_move = -1
        self.move_count = 0
        self._chain_pos = None

        self.board[:, :] = EMPTY

        # P1 bottom rows 0..2
        for r in range(3):
            for c in range(8):
                if (r + c) % 2 == 1:
                    self.board[r, c] = P1_MAN
        # P2 top rows 5..7
        for r in range(5, 8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    self.board[r, c] = P2_MAN

        self.availables = self.get_legal_actions()
        self.states = {}

    def _count_pieces(self):
        p1 = int(np.sum((self.board == P1_MAN) | (self.board == P1_KING)))
        p2 = int(np.sum((self.board == P2_MAN) | (self.board == P2_KING)))
        return p1, p2

    def _piece_belongs_to_player(self, piece: int, player: int) -> bool:
        return _is_p1(piece) if player == 1 else _is_p2(piece)

    def _is_opponent_piece(self, piece: int, player: int) -> bool:
        if piece == EMPTY:
            return False
        return _is_p2(piece) if player == 1 else _is_p1(piece)

    def current_state(self):
        """
        5x8x8 state from current player's perspective:
        0: current men
        1: current kings
        2: opponent men
        3: opponent kings
        4: to_play_is_p1 (all ones if current_player==1 else zeros)
        """
        s = np.zeros((6, 8, 8), dtype=np.float32)
        if self.current_player == 1:
            s[0][self.board == P1_MAN] = 1.0
            s[1][self.board == P1_KING] = 1.0
            s[2][self.board == P2_MAN] = 1.0
            s[3][self.board == P2_KING] = 1.0
            s[4][:, :] = 1.0
        else:
            s[0][self.board == P2_MAN] = 1.0
            s[1][self.board == P2_KING] = 1.0
            s[2][self.board == P1_MAN] = 1.0
            s[3][self.board == P1_KING] = 1.0
            s[4][:, :] = 0.0
        #  连吃续走的棋子位置
        if self._chain_pos is not None:
            r, c = self._chain_pos
            s[5, r, c] = 1.0
    
        return s

    def _dirs_for_piece(self, piece: int, player: int):
        if _is_king(piece):
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        # men: forward only :contentReference[oaicite:2]{index=2}
        return [(1, -1), (1, 1)] if player == 1 else [(-1, -1), (-1, 1)]

    def _simple_moves_from(self, r: int, c: int, piece: int, player: int):
        moves = []
        for dr, dc in self._dirs_for_piece(piece, player):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8 and (rr + cc) % 2 == 1 and self.board[rr, cc] == EMPTY:
                f = int(self._rc_to_dark[r, c])
                t = int(self._rc_to_dark[rr, cc])
                moves.append(f * 32 + t)
        return moves

    def _captures_from(self, r: int, c: int, piece: int, player: int):
        caps = []
        for dr, dc in self._dirs_for_piece(piece, player):
            mr, mc = r + dr, c + dc
            rr, cc = r + 2 * dr, c + 2 * dc
            if 0 <= rr < 8 and 0 <= cc < 8 and 0 <= mr < 8 and 0 <= mc < 8:
                if (rr + cc) % 2 != 1:
                    continue
                if self._is_opponent_piece(int(self.board[mr, mc]), player) and self.board[rr, cc] == EMPTY:
                    f = int(self._rc_to_dark[r, c])
                    t = int(self._rc_to_dark[rr, cc])
                    caps.append(f * 32 + t)
        return caps

    def get_legal_actions(self):
        # forced continuation
        if self._chain_pos is not None:
            r, c = self._chain_pos
            piece = int(self.board[r, c])
            if piece != EMPTY and self._piece_belongs_to_player(piece, self.current_player):
                return self._captures_from(r, c, piece, self.current_player)
            self._chain_pos = None

        capture_moves = []
        for di in range(32):
            r, c = self._dark_to_rc[di]
            piece = int(self.board[r, c])
            if piece != EMPTY and self._piece_belongs_to_player(piece, self.current_player):
                capture_moves.extend(self._captures_from(r, c, piece, self.current_player))
        if capture_moves:
            return capture_moves

        legal = []
        for di in range(32):
            r, c = self._dark_to_rc[di]
            piece = int(self.board[r, c])
            if piece != EMPTY and self._piece_belongs_to_player(piece, self.current_player):
                legal.extend(self._simple_moves_from(r, c, piece, self.current_player))
        return legal

    def do_move(self, move: int):
        if move < 0 or move >= self.action_size:
            raise ValueError("Invalid action id")

        if move not in self.availables:
            raise ValueError("Move not legal in current position")

        from_idx = move // 32
        to_idx = move % 32
        fr, fc = self._dark_to_rc[from_idx]
        tr, tc = self._dark_to_rc[to_idx]

        piece = int(self.board[fr, fc])
        self.board[fr, fc] = EMPTY

        dr, dc = tr - fr, tc - fc
        is_capture = (abs(dr) == 2 and abs(dc) == 2)

        # place
        self.board[tr, tc] = piece

        # remove captured
        if is_capture:
            mr, mc = fr + dr // 2, fc + dc // 2
            self.board[mr, mc] = EMPTY

        # promotion
        promoted = False
        if piece == P1_MAN and tr == 7:
            self.board[tr, tc] = P1_KING
            piece = P1_KING
            promoted = True
        elif piece == P2_MAN and tr == 0:
            self.board[tr, tc] = P2_KING
            piece = P2_KING
            promoted = True

        self.last_move = int(move)
        self.move_count += 1

        # if promoted and rule says turn ends now  :contentReference[oaicite:3]{index=3}
        if promoted and self.promote_ends_turn:
            self._chain_pos = None
            self.current_player = 1 if self.current_player == 2 else 2
            self.availables = self.get_legal_actions()
            return

        # multi-capture continuation
        if is_capture:
            more_caps = self._captures_from(tr, tc, piece, self.current_player)
            if more_caps:
                self._chain_pos = (tr, tc)
                self.availables = more_caps
                return

        # normal switch
        self._chain_pos = None
        self.current_player = 1 if self.current_player == 2 else 2
        self.availables = self.get_legal_actions()

    def game_end(self):
        p1, p2 = self._count_pieces()
        if p1 == 0:
            return True, 2
        if p2 == 0:
            return True, 1

        legal = self.availables if self.availables is not None else self.get_legal_actions()
        if len(legal) == 0:
            winner = 1 if self.current_player == 2 else 2
            return True, winner

        if self.move_count >= self.max_moves:
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        print("Player", player1, "with w/W".rjust(8))
        print("Player", player2, "with b/B".rjust(8))
        print("Input: from_dark,to_dark (0..31). Example: 12,16")
        print("----")

        print("    " + " ".join([str(c) for c in range(8)]))
        for r in range(7, -1, -1):
            row = [str(r).rjust(2)]
            for c in range(8):
                piece = int(board.board[r, c])
                if (r + c) % 2 == 0:
                    row.append(" ")
                else:
                    row.append({EMPTY: ".", P1_MAN: "w", P1_KING: "W", P2_MAN: "b", P2_KING: "B"}.get(piece, "?"))
            print(" ".join(row))
        print("Legal moves:", len(board.availables))
        if board._chain_pos is not None:
            r, c = board._chain_pos
            di = int(board._rc_to_dark[r, c])
            print(">> Must continue multi-capture with piece at dark index:", di)
        print()

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        while True:
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

            current_player = self.board.get_current_player()
            move = players[current_player].get_action(self.board)
            self.board.do_move(move)

            if is_shown:
                self.graphic(self.board, player1.player, player2.player)

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []

        while True:
            end, winner = self.board.game_end()
            if end:
                winners_z = np.zeros(len(current_players), dtype=np.float32)
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                return winner, zip(states, mcts_probs, winners_z)

            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
