# -*- coding: utf-8 -*-
from datetime import datetime, timezone
from db import users_col

def _now():
    return datetime.now(timezone.utc)

def _mode_key(mode: str) -> str:
    # 前端：pvp_net / pve / pvp_local
    # 只存 pvp / pve
    if mode == "pve":
        return "pve"
    return "pvp"

def record_match_for_user(username: str, game: str, mode: str, result: str, match_summary: dict, keep_last: int = 50):
    """
    result: 'win'|'lose'|'draw'
    match_summary: 会被 push 到 users.matches（按账号集中存）
    """
    g = "gomoku" if game == "gomoku" else "checkers"
    mk = _mode_key(mode)

    inc = {
        f"stats.{g}.{mk}.games": 1,
        f"stats.{g}.{mk}.{result}": 1
    }

    match_doc = {
        "ts": _now(),
        "game": g,
        "mode": mk,
        "result": result,
        **match_summary
    }

    users_col().update_one(
        {"username": username},
        {
            "$inc": inc,
            "$push": {"matches": {"$each": [match_doc], "$slice": -keep_last}}
        }
    )

def record_pvp_result(game: str, winner: int, p1: str, p2: str, mode: str, extra: dict):
    # winner: 1/2/-1
    if winner == -1:
        record_match_for_user(p1, game, mode, "draw", {"opponent": p2, "winner": -1, **extra})
        record_match_for_user(p2, game, mode, "draw", {"opponent": p1, "winner": -1, **extra})
        return
    if winner == 1:
        record_match_for_user(p1, game, mode, "win",  {"opponent": p2, "winner": 1, **extra})
        record_match_for_user(p2, game, mode, "lose", {"opponent": p1, "winner": 1, **extra})
        return
    if winner == 2:
        record_match_for_user(p1, game, mode, "lose", {"opponent": p2, "winner": 2, **extra})
        record_match_for_user(p2, game, mode, "win",  {"opponent": p1, "winner": 2, **extra})
        return

def record_pve_result(game: str, winner: int, human_player: int, username: str, mode: str, extra: dict):
    # winner: 1/2/-1 ； human_player: 1/2
    if winner == -1:
        record_match_for_user(username, game, mode, "draw", {"opponent": "AI", "winner": -1, **extra})
        return
    if winner == human_player:
        record_match_for_user(username, game, mode, "win",  {"opponent": "AI", "winner": winner, **extra})
    else:
        record_match_for_user(username, game, mode, "lose", {"opponent": "AI", "winner": winner, **extra})