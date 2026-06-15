# games/checkers_rules.py
# -*- coding: utf-8 -*-
from games.checker_game import EMPTY, P1_MAN, P2_MAN, P1_KING, P2_KING, Board

ACTION_SIZE = 32 * 32

def new_board(max_moves=200, promote_ends_turn=True) -> Board:
    b = Board(width=8, height=8, max_moves=max_moves, promote_ends_turn=promote_ends_turn)
    b.init_board(start_player=0)
    return b

def board_to_list64(b: Board):
    return [int(x) for x in b.board.reshape(-1).tolist()]

def engine_to_action(b: Board, frm_engine: int, to_engine: int) -> int:
    r1, c1 = int(frm_engine)//8, int(frm_engine)%8
    r2, c2 = int(to_engine)//8, int(to_engine)%8
    fd = int(b._rc_to_dark[r1, c1])
    td = int(b._rc_to_dark[r2, c2])
    if fd < 0 or td < 0:
        return -1
    return fd * 32 + td