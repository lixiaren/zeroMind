# -*- coding: utf-8 -*-
"""Lightweight tactical helpers for checkers agents."""

from __future__ import annotations

import copy
from typing import Optional

from games.checker_game import EMPTY, P1_KING, P1_MAN, P2_KING, P2_MAN


PIECE_VALUE = {
    EMPTY: 0,
    P1_MAN: 100,
    P2_MAN: 100,
    P1_KING: 175,
    P2_KING: 175,
}


def action_points(board, action: int):
    from_idx = int(action) // 32
    to_idx = int(action) % 32
    fr, fc = board._dark_to_rc[from_idx]
    tr, tc = board._dark_to_rc[to_idx]
    return fr, fc, tr, tc


def is_capture(board, action: int) -> bool:
    fr, fc, tr, tc = action_points(board, action)
    return abs(tr - fr) == 2 and abs(tc - fc) == 2


def _captured_value(board, action: int) -> int:
    if not is_capture(board, action):
        return 0
    fr, fc, tr, tc = action_points(board, action)
    mr, mc = fr + (tr - fr) // 2, fc + (tc - fc) // 2
    return PIECE_VALUE.get(int(board.board[mr, mc]), 0)


def _promotes(board, action: int) -> bool:
    fr, fc, tr, tc = action_points(board, action)
    piece = int(board.board[fr, fc])
    return (piece == P1_MAN and tr == 7) or (piece == P2_MAN and tr == 0)


def _opponent_positions(board, player: int):
    opponent_pieces = (P2_MAN, P2_KING) if int(player) == 1 else (P1_MAN, P1_KING)
    positions = []
    for r in range(8):
        for c in range(8):
            if int(board.board[r, c]) in opponent_pieces:
                positions.append((r, c))
    return positions


def _nearest_distance(row: int, col: int, positions) -> float:
    if not positions:
        return 0.0
    return float(min(max(abs(row - r), abs(col - c)) for r, c in positions))


def _is_player_piece(piece: int, player: int) -> bool:
    if int(player) == 1:
        return int(piece) in (P1_MAN, P1_KING)
    return int(piece) in (P2_MAN, P2_KING)


def _material_balance(board, player: int) -> int:
    own = 0
    opp = 0
    for r in range(8):
        for c in range(8):
            value = PIECE_VALUE.get(int(board.board[r, c]), 0)
            if value <= 0:
                continue
            if _is_player_piece(int(board.board[r, c]), player):
                own += value
            else:
                opp += value
    return own - opp


def _landing_is_capturable(next_board, landing_dark: int, moving_player: int) -> bool:
    if int(next_board.current_player) == int(moving_player):
        return False

    lr, lc = next_board._dark_to_rc[int(landing_dark)]
    for reply in list(next_board.availables or []):
        if not is_capture(next_board, reply):
            continue
        fr, fc, tr, tc = action_points(next_board, reply)
        mr, mc = fr + (tr - fr) // 2, fc + (tc - fc) // 2
        if (mr, mc) == (lr, lc):
            return True
    return False


def action_outcome(board, action: int):
    """Return immediate material gain and worst opponent capture cost."""
    player = int(board.current_player)
    before = _material_balance(board, player)

    next_board = copy.deepcopy(board)
    next_board.do_move(int(action))
    after = _material_balance(next_board, player)
    gain = float(after - before)

    # Same player means forced multi-capture continuation, not an opponent reply.
    if int(next_board.current_player) == player:
        return gain, 0.0, False

    reply_cost = 0.0
    has_capture_reply = False
    for reply in list(next_board.availables or []):
        if not is_capture(next_board, reply):
            continue
        has_capture_reply = True
        reply_board = copy.deepcopy(next_board)
        reply_board.do_move(int(reply))
        reply_after = _material_balance(reply_board, player)
        reply_cost = max(reply_cost, float(after - reply_after))

    return gain, reply_cost, has_capture_reply


def is_no_benefit_sacrifice(board, action: int, margin: float = 15.0) -> bool:
    """True when the move gives the opponent a capture without enough gain."""
    try:
        gain, reply_cost, has_capture_reply = action_outcome(board, action)
    except Exception:
        return False
    if not has_capture_reply:
        return False
    return (gain - reply_cost) < -float(margin)


def action_score(board, action: int) -> float:
    player = int(board.current_player)
    fr, fc, tr, tc = action_points(board, action)
    piece = int(board.board[fr, fc])

    score = 0.0
    if is_capture(board, action):
        score += 1_000.0 + _captured_value(board, action)

    if _promotes(board, action):
        score += 350.0

    # Prefer central landing squares and steady development for men.
    score += (3.5 - abs(tc - 3.5)) * 8.0
    if piece == P1_MAN:
        score += tr * 6.0
    elif piece == P2_MAN:
        score += (7 - tr) * 6.0
    elif piece in (P1_KING, P2_KING):
        opponents = _opponent_positions(board, player)
        before = _nearest_distance(fr, fc, opponents)
        after = _nearest_distance(tr, tc, opponents)

        # Kings should hunt. Without this, a weak net can keep sliding on the
        # promotion row because "safe" sideways moves look acceptable.
        score += (before - after) * 55.0
        score += (3.5 - abs(tr - 3.5)) * 8.0
        if after <= 2:
            score += 35.0
        if tr in (0, 7):
            score -= 45.0
        if fr in (0, 7) and tr not in (0, 7):
            score += 55.0

    try:
        gain, reply_cost, has_capture_reply = action_outcome(board, action)
        next_board = copy.deepcopy(board)
        next_board.do_move(int(action))
        if int(next_board.current_player) == player:
            # Multi-capture continuation: more legal captures from this piece is good.
            score += 180.0 * len(next_board.availables or [])

        # Strongly punish "move there and get eaten" patterns. The earlier
        # version only applied a light landing penalty, which let the AI trade
        # away pieces for no real compensation.
        if has_capture_reply:
            net_after_reply = gain - reply_cost
            score += net_after_reply * 2.0
            if net_after_reply < -15.0:
                score -= 700.0 + reply_cost * 3.0
            elif net_after_reply < 20.0 and not is_capture(board, action):
                score -= 250.0 + reply_cost

        if _landing_is_capturable(next_board, int(action) % 32, player):
            score -= PIECE_VALUE.get(piece, 100) * 1.8
    except Exception:
        pass

    return score


def select_tactical_action(board) -> Optional[int]:
    legal = list(getattr(board, "availables", []) or [])
    if not legal:
        return None
    if len(legal) == 1:
        return int(legal[0])

    captures = [action for action in legal if is_capture(board, action)]
    if captures:
        return int(max(captures, key=lambda action: (action_score(board, action), -int(action))))

    promotions = [action for action in legal if _promotes(board, action)]
    if promotions:
        safe_promotions = [action for action in promotions if not is_no_benefit_sacrifice(board, action)]
        candidates = safe_promotions if safe_promotions else promotions
        return int(max(candidates, key=lambda action: (action_score(board, action), -int(action))))

    return None


def select_fallback_action(board) -> Optional[int]:
    legal = list(getattr(board, "availables", []) or [])
    if not legal:
        return None

    safe = [action for action in legal if not is_no_benefit_sacrifice(board, action)]
    candidates = safe if safe else legal
    return int(max(candidates, key=lambda action: (action_score(board, action), -int(action))))
