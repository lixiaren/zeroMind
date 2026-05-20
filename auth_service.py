# -*- coding: utf-8 -*-
import secrets
from datetime import datetime, timedelta, timezone
from pymongo.errors import DuplicateKeyError

from config import JWT_EXPIRE_DAYS  # 你已有的配置，继续拿来当“会话有效期(天)”用
from db import users_col


def _now():
    return datetime.now(timezone.utc)


def _col():
    # 兼容：users_col 可能是函数，也可能是 collection
    return users_col() if callable(users_col) else users_col


def _issue_token():
    # 随机字符串 token（不是 JWT）
    return secrets.token_urlsafe(32)






def _ensure_utc_aware(dt):
    if not isinstance(dt, datetime):
        return None
    # Mongo 读出来经常是 naive，但语义是 UTC
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def verify_token(token: str):
    token = (token or "").strip()
    if not token:
        return None

    u = users_col().find_one({"session.token": token}, {"username": 1, "session": 1})
    if not u:
        return None

    exp = _ensure_utc_aware(u.get("session", {}).get("exp"))
    if not exp:
        return None

    if exp < _now():
        return None

    return u.get("username")


def register_user(username: str, password: str) -> dict:
    username = (username or "").strip()
    password = password or ""

    if len(username) < 3:
        return {"ok": False, "msg": "Username is too short (at least 3 characters)."}
    if len(password) < 1:
        return {"ok": False, "msg": "Password cannot be empty."}

    token = _issue_token()
    exp = _now() + timedelta(days=int(JWT_EXPIRE_DAYS))

    doc = {
        "username": username,
        "password": password,  # ✅ 明文存（你要求不在乎安全）
        "created_at": _now(),
        "last_login": _now(),
        "session": {"token": token, "exp": exp},

        # 一个账号一份文档：所有统计/战绩都塞这里
        "stats": {
            "gomoku": {
                "pvp": {"games": 0, "win": 0, "lose": 0, "draw": 0},
                "pve": {"games": 0, "win": 0, "lose": 0, "draw": 0},
            },
            "checkers": {
                "pvp": {"games": 0, "win": 0, "lose": 0, "draw": 0},
                "pve": {"games": 0, "win": 0, "lose": 0, "draw": 0},
            },
        },
        "matches": []
    }

    try:
        _col().insert_one(doc)
    except DuplicateKeyError:
        return {"ok": False, "msg": "Username already exists."}

    return {"ok": True, "token": token, "user": {"username": username}}


def login_user(username: str, password: str) -> dict:
    username = (username or "").strip()
    password = password or ""

    u = _col().find_one({"username": username})
    if not u:
        return {"ok": False, "msg": "User not found."}

    if (u.get("password") or "") != password:
        return {"ok": False, "msg": "Wrong password."}

    token = _issue_token()
    exp = _now() + timedelta(days=int(JWT_EXPIRE_DAYS))

    _col().update_one(
        {"_id": u["_id"]},
        {"$set": {"last_login": _now(), "session": {"token": token, "exp": exp}}}
    )

    return {"ok": True, "token": token, "user": {"username": username}}
