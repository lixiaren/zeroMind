# -*- coding: utf-8 -*-
from db import users_col

def _winrate(s: dict) -> float:
    g = int(s.get("games", 0))
    if g <= 0:
        return 0.0
    return round(100.0 * (int(s.get("win", 0)) / g), 2)

def get_profile(username: str) -> dict:
    u = users_col().find_one({"username": username}, {"_id": 0, "pw_hash": 0})
    if not u:
        return {"ok": False, "msg": "User not found."}

    stats = u.get("stats", {})
    # 计算 winrate（不额外存，避免冗余）
    wr = {
        "gomoku": {
            "pvp": _winrate(stats.get("gomoku", {}).get("pvp", {})),
            "pve": _winrate(stats.get("gomoku", {}).get("pve", {})),
        },
        "checkers": {
            "pvp": _winrate(stats.get("checkers", {}).get("pvp", {})),
            "pve": _winrate(stats.get("checkers", {}).get("pve", {})),
        },
    }

    return {
        "ok": True,
        "user": {"username": u["username"]},
        "stats": stats,
        "winrate": wr,
        "matches": u.get("matches", [])[-20:]  # 给前端看最近 20 条
    }
