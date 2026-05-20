# -*- coding: utf-8 -*-
try:
    import eventlet
    eventlet.monkey_patch()
    SOCKET_ASYNC_MODE = "eventlet"
except ModuleNotFoundError:
    SOCKET_ASYNC_MODE = "threading"

import random
import uuid
import  torch
from flask import Flask, request, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room

from config import JWT_SECRET, SOCKET_CORS
from db import init_indexes, users_col, chats_col
from auth_service import register_user, login_user, verify_token
from profile_service import get_profile
from stats_service import record_pvp_result, record_pve_result
from matchmaker import Matchmaker
from sessions.gomoku_session import GomokuSession
from sessions.checkers_session import CheckersSession
from datetime import datetime, date, timezone

try:
    from bson import ObjectId
except Exception:
    ObjectId = None


def _jsonable(x):
    """递归把 datetime/ObjectId 等转成 JSON 可序列化对象"""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if ObjectId is not None and isinstance(x, ObjectId):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    # 兜底：别再让 emit 崩
    return str(x)

import os
from ai.gomoku_mcts_agent import GomokuPolicyPool, GomokuAZAgent

GOMOKU_BOARD_SIZE = int(os.getenv("GOMOKU_BOARD_SIZE", "15"))
GOMOKU_N_IN_ROW = int(os.getenv("GOMOKU_N_IN_ROW", "5"))
GOMOKU_MODEL_FILE = os.getenv("GOMOKU_MODEL_FILE", "best_policy15.model")
GOMOKU_USE_GPU = os.getenv("GOMOKU_USE_GPU", "1") == "1"
GOMOKU_N_PLAYOUT = int(os.getenv("GOMOKU_N_PLAYOUT", "50"))
GOMOKU_C_PUCT = float(os.getenv("GOMOKU_C_PUCT", "5"))
GOMOKU_SKIP_ATTENTION = os.getenv("GOMOKU_SKIP_ATTENTION", os.getenv("SKIP_ATTENTION", "0")) == "1"
GOMOKU_MCTS_BATCH_SIZE = int(os.getenv("GOMOKU_MCTS_BATCH_SIZE", os.getenv("MCTS_BATCH_SIZE", "8")))
GOMOKU_TACTICS_MODE = os.getenv("GOMOKU_TACTICS_MODE", "emergency")

from ai.checkers_mcts_agent import CheckersPolicyPool, CheckersAZAgent

CHECKERS_MODEL_FILE = os.getenv("CHECKERS_MODEL_FILE", "best_checker_policy.model")
CHECKERS_USE_GPU = os.getenv("CHECKERS_USE_GPU", "1") == "1"
CHECKERS_N_PLAYOUT = int(os.getenv("CHECKERS_N_PLAYOUT", "50"))
CHECKERS_C_PUCT = float(os.getenv("CHECKERS_C_PUCT", "5"))
CHECKERS_SKIP_ATTENTION = os.getenv("CHECKERS_SKIP_ATTENTION", os.getenv("SKIP_ATTENTION", "0")) == "1"
CHECKERS_MCTS_BATCH_SIZE = int(os.getenv("CHECKERS_MCTS_BATCH_SIZE", os.getenv("MCTS_BATCH_SIZE", "8")))
CHECKERS_SAFETY_GAP = float(os.getenv("CHECKERS_SAFETY_GAP", "60"))
CHECKERS_TACTICS_MODE = os.getenv("CHECKERS_TACTICS_MODE", "off")

AI_DIFFICULTIES = ("easy", "normal", "hard")
AI_DIFFICULTY_LABELS = {
    "easy": "Easy",
    "normal": "Normal",
    "hard": "Hard",
}
GOMOKU_DIFFICULTY_CONFIG = {
    "easy": {
        "model": os.getenv("GOMOKU_EASY_MODEL_FILE", "gomoku15_policy_0500.model"),
        "playouts": int(os.getenv("GOMOKU_EASY_N_PLAYOUT", "20")),
        "c_puct": float(os.getenv("GOMOKU_EASY_C_PUCT", str(GOMOKU_C_PUCT))),
        "tactics": os.getenv("GOMOKU_EASY_TACTICS_MODE", GOMOKU_TACTICS_MODE),
    },
    "normal": {
        "model": os.getenv("GOMOKU_NORMAL_MODEL_FILE", GOMOKU_MODEL_FILE),
        "playouts": int(os.getenv("GOMOKU_NORMAL_N_PLAYOUT", str(GOMOKU_N_PLAYOUT))),
        "c_puct": float(os.getenv("GOMOKU_NORMAL_C_PUCT", str(GOMOKU_C_PUCT))),
        "tactics": os.getenv("GOMOKU_NORMAL_TACTICS_MODE", GOMOKU_TACTICS_MODE),
    },
    "hard": {
        "model": os.getenv("GOMOKU_HARD_MODEL_FILE", GOMOKU_MODEL_FILE),
        "playouts": int(os.getenv("GOMOKU_HARD_N_PLAYOUT", "120")),
        "c_puct": float(os.getenv("GOMOKU_HARD_C_PUCT", str(GOMOKU_C_PUCT))),
        "tactics": os.getenv("GOMOKU_HARD_TACTICS_MODE", "full"),
    },
}
CHECKERS_DIFFICULTY_CONFIG = {
    "easy": {
        "model": os.getenv("CHECKERS_EASY_MODEL_FILE", "checker_policy_0500.model"),
        "playouts": int(os.getenv("CHECKERS_EASY_N_PLAYOUT", "20")),
        "c_puct": float(os.getenv("CHECKERS_EASY_C_PUCT", str(CHECKERS_C_PUCT))),
        "tactics": os.getenv("CHECKERS_EASY_TACTICS_MODE", CHECKERS_TACTICS_MODE),
    },
    "normal": {
        "model": os.getenv("CHECKERS_NORMAL_MODEL_FILE", CHECKERS_MODEL_FILE),
        "playouts": int(os.getenv("CHECKERS_NORMAL_N_PLAYOUT", str(CHECKERS_N_PLAYOUT))),
        "c_puct": float(os.getenv("CHECKERS_NORMAL_C_PUCT", str(CHECKERS_C_PUCT))),
        "tactics": os.getenv("CHECKERS_NORMAL_TACTICS_MODE", CHECKERS_TACTICS_MODE),
    },
    "hard": {
        "model": os.getenv("CHECKERS_HARD_MODEL_FILE", CHECKERS_MODEL_FILE),
        "playouts": int(os.getenv("CHECKERS_HARD_N_PLAYOUT", "120")),
        "c_puct": float(os.getenv("CHECKERS_HARD_C_PUCT", str(CHECKERS_C_PUCT))),
        "tactics": os.getenv("CHECKERS_HARD_TACTICS_MODE", "full"),
    },
}
GOMOKU_POOLS = {}
CHECKERS_POOLS = {}


def _normalize_difficulty(value):
    difficulty = str(value or "normal").strip().lower()
    return difficulty if difficulty in AI_DIFFICULTIES else "normal"


def _existing_model_or_fallback(game, difficulty, model_file, fallback_file):
    if os.path.exists(model_file):
        return model_file
    print("[{}] {} model missing: {}. fallback -> {}".format(game.upper(), difficulty, model_file, fallback_file))
    return fallback_file


def _get_gomoku_pool(difficulty):
    difficulty = _normalize_difficulty(difficulty)
    cfg = GOMOKU_DIFFICULTY_CONFIG[difficulty]
    normal_model = GOMOKU_DIFFICULTY_CONFIG["normal"]["model"]
    fallback = normal_model if os.path.exists(normal_model) else GOMOKU_MODEL_FILE
    model_file = _existing_model_or_fallback("gomoku", difficulty, cfg["model"], fallback)
    cache_key = (difficulty, model_file, GOMOKU_BOARD_SIZE)
    if cache_key not in GOMOKU_POOLS:
        GOMOKU_POOLS[cache_key] = GomokuPolicyPool(
            GOMOKU_BOARD_SIZE,
            GOMOKU_BOARD_SIZE,
            model_file=model_file,
            use_gpu=GOMOKU_USE_GPU,
            skip_attention=GOMOKU_SKIP_ATTENTION,
        )
    return GOMOKU_POOLS[cache_key], cfg


def _get_checkers_pool(difficulty):
    difficulty = _normalize_difficulty(difficulty)
    cfg = CHECKERS_DIFFICULTY_CONFIG[difficulty]
    normal_model = CHECKERS_DIFFICULTY_CONFIG["normal"]["model"]
    fallback = normal_model if os.path.exists(normal_model) else CHECKERS_MODEL_FILE
    model_file = _existing_model_or_fallback("checkers", difficulty, cfg["model"], fallback)
    cache_key = (difficulty, model_file)
    if cache_key not in CHECKERS_POOLS:
        CHECKERS_POOLS[cache_key] = CheckersPolicyPool(
            model_file=model_file,
            use_gpu=CHECKERS_USE_GPU,
            skip_attention=CHECKERS_SKIP_ATTENTION,
        )
    return CHECKERS_POOLS[cache_key], cfg

print("[GOMOKU] cwd =", os.getcwd())
print("[GOMOKU] board =", "{}x{}, n_in_row={}".format(GOMOKU_BOARD_SIZE, GOMOKU_BOARD_SIZE, GOMOKU_N_IN_ROW))
print("[GOMOKU] normal model_file =", GOMOKU_DIFFICULTY_CONFIG["normal"]["model"])
print("[GOMOKU] normal exists =", os.path.exists(GOMOKU_DIFFICULTY_CONFIG["normal"]["model"]))
print("[CHECKERS] normal model_file =", CHECKERS_DIFFICULTY_CONFIG["normal"]["model"])
print("[CHECKERS] normal exists =", os.path.exists(CHECKERS_DIFFICULTY_CONFIG["normal"]["model"]))
print("[GOMOKU] torch cuda available =", torch.cuda.is_available())
print("[GOMOKU] use_gpu =", GOMOKU_USE_GPU)
print("[GOMOKU] skip_attention =", GOMOKU_SKIP_ATTENTION)
print("[CHECKERS] skip_attention =", CHECKERS_SKIP_ATTENTION)
print("[CHECKERS] safety_gap =", CHECKERS_SAFETY_GAP)
print("[CHECKERS] normal tactics =", CHECKERS_DIFFICULTY_CONFIG["normal"]["tactics"])

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=SOCKET_CORS, async_mode=SOCKET_ASYNC_MODE)


@app.route("/")
def index():
    return send_file(os.path.join(os.path.dirname(__file__), "game_ui_2.html"), max_age=0)


@app.route("/health")
def health():
    return {
        "ok": True,
        "cuda": bool(torch.cuda.is_available()),
        "gomokuBoardSize": GOMOKU_BOARD_SIZE,
        "gomokuNInRow": GOMOKU_N_IN_ROW,
        "gomokuDifficultyModels": {
            k: v["model"] for k, v in GOMOKU_DIFFICULTY_CONFIG.items()
        },
        "gomokuDifficultyTactics": {
            k: v["tactics"] for k, v in GOMOKU_DIFFICULTY_CONFIG.items()
        },
        "checkersDifficultyModels": {
            k: v["model"] for k, v in CHECKERS_DIFFICULTY_CONFIG.items()
        },
        "checkersDifficultyTactics": {
            k: v["tactics"] for k, v in CHECKERS_DIFFICULTY_CONFIG.items()
        },
    }

init_indexes()
mm = Matchmaker()

# matchId -> session
SESSIONS = {}

# challengeId -> {from,to,game}
CHALLENGES = {}

SUPPORTED_GAMES = {"gomoku", "checkers"}


# -------------------------
# Social helpers
# -------------------------
def _now_utc():
    return datetime.now(timezone.utc)


def _clean_username(value):
    return (value or "").strip()


def _get_user(username, projection=None):
    username = _clean_username(username)
    if not username:
        return None
    return users_col().find_one({"username": username}, projection)


def _user_exists(username):
    return _get_user(username, {"_id": 1}) is not None


def _relation_between(me, username):
    if not me or not username:
        return "none"
    if me == username:
        return "self"

    doc = _get_user(me, {"friends": 1, "friend_requests_in": 1, "friend_requests_out": 1})
    if not doc:
        return "none"

    if username in (doc.get("friends") or []):
        return "friends"
    if username in (doc.get("friend_requests_in") or []):
        return "incoming"
    if username in (doc.get("friend_requests_out") or []):
        return "outgoing"
    return "none"


def _friend_entry(username):
    return {"username": username, "online": bool(mm.is_online(username))}


def _social_state(username):
    doc = _get_user(username, {"friends": 1, "friend_requests_in": 1, "friend_requests_out": 1})
    if not doc:
        return {"ok": False, "msg": "User not found"}

    friends = sorted(set(doc.get("friends") or []), key=str.lower)
    incoming = sorted(set(doc.get("friend_requests_in") or []), key=str.lower)
    outgoing = sorted(set(doc.get("friend_requests_out") or []), key=str.lower)
    return {
        "ok": True,
        "friends": [_friend_entry(u) for u in friends if _user_exists(u)],
        "incoming": [_friend_entry(u) for u in incoming if _user_exists(u)],
        "outgoing": [_friend_entry(u) for u in outgoing if _user_exists(u)],
    }


def _emit_social_state(username):
    sid = mm.sid_of(username)
    if sid:
        socketio.emit("friends_state", _jsonable(_social_state(username)), to=sid)


def _emit_social_pair(a, b):
    _emit_social_state(a)
    _emit_social_state(b)


def _broadcast_presence(username):
    doc = _get_user(username, {"friends": 1})
    if not doc:
        return
    payload = {"username": username, "online": bool(mm.is_online(username))}
    for friend in doc.get("friends") or []:
        fsid = mm.sid_of(friend)
        if fsid:
            socketio.emit("presence_update", payload, to=fsid)
            _emit_social_state(friend)


def _conversation_id(a, b):
    return "|".join(sorted([a, b], key=str.lower))


def _are_friends(a, b):
    doc = _get_user(a, {"friends": 1})
    return bool(doc and b in (doc.get("friends") or []))


# -------------------------
# Helpers
# -------------------------
def _sid():
    return request.sid


def _me():
    return mm.username_of(_sid())


def _need_login():
    me = _me()
    if not me:
        emit("error", {"msg": "Not signed in."})
        return None
    return me


def _get_match_id(data):
    if isinstance(data, dict):
        mid = data.get("matchId")
        if mid:
            return mid
    return mm.sid_match.get(_sid())


def _get_session_or_err(match_id):
    sess = SESSIONS.get(match_id)
    if not sess:
        emit("error", {"msg": "Match not found or already ended."})
        return None
    return sess


def _your_player(sess, me):
    """
    返回你在此对局中的 player 编号（1/2），否则 None
    - pvp_net: p1=player1, p2=player2
    - pve:
        gomoku: sess.human_side = 1/2
        checkers: sess.human_player = 1/2
    """
    if sess.mode == "local":
        # 本地自对弈：你操控双方，按后端 current_player 来
        return int(getattr(sess, "current_player", 1))
    if sess.mode == "pvp_net":
        if me == sess.p1:
            return 1
        if me == sess.p2:
            return 2
        return None

    # pve：human 必须是 sess.p1
    if me != sess.p1:
        return None

    if sess.game == "gomoku":
        return int(getattr(sess, "human_side", 1))
    return int(getattr(sess, "human_player", 1))


def _emit_state(sess):
    # ✅ Local：前端 myPlayer 只在 match_found 更新一次
    #    为了不改前端，让后端每次同步都刷新“当前该谁走”
    if sess.mode == "local":
        cp = int(getattr(sess, "current_player", 1))
        for psid in getattr(sess, "sids", []):
            socketio.emit(
                "match_found",
                {"matchId": sess.match_id, "game": sess.game, "mode": "local", "yourPlayer": cp, "opponent": "Local"},
                to=psid,
            )
    socketio.emit("sync_state", sess.state_payload(), to=sess.room)
    socketio.sleep(0)  # ✅ 关键：让 eventlet 立刻把包发出去

def _cleanup_session(sess):
    # 清房间、清 match 标记、删内存
    for psid in getattr(sess, "sids", []):
        try:
            mm.clear_match(psid)
        except Exception:
            pass
        try:
            leave_room(sess.room, sid=psid)
        except Exception:
            pass
    SESSIONS.pop(sess.match_id, None)
def _detach_sid(sess, sid: str):
    # 退出者先脱离房间 + 清理 match 标记，避免收到后续 game_over
    try:
        mm.clear_match(sid)
    except Exception:
        pass
    try:
        leave_room(sess.room, sid=sid)
    except Exception:
        pass
    try:
        if hasattr(sess, "sids") and sid in sess.sids:
            sess.sids.remove(sid)
    except Exception:
        pass

def _record_result(sess):
    if sess.mode == "local":
        return
    # 战绩写库
    if sess.mode == "pvp_net":
        record_pvp_result(
            sess.game,
            sess.winner,
            sess.p1,
            sess.p2,
            sess.mode,
            {"matchId": sess.match_id},
        )
        return

    # pve
    if sess.game == "gomoku":
        record_pve_result(
            sess.game,
            sess.winner,
            sess.human_side,
            sess.p1,
            sess.mode,
            {"matchId": sess.match_id},
        )
    else:
        record_pve_result(
            sess.game,
            sess.winner,
            sess.human_player,
            sess.p1,
            sess.mode,
            {"matchId": sess.match_id},
        )


def _finalize(sess):
    
    # 广播 game_over + 写库 + 清理
    socketio.emit(
        "game_over",
        {"winner": sess.winner, "matchId": sess.match_id, "game": sess.game},
        to=sess.room,
    )
    
    _record_result(sess)
    _cleanup_session(sess)


def _ai_play_until_human(sess, max_steps=64):
    """
    PvE：让 AI 走到轮到 human 或结束。
    不同游戏的 ai_move_if_needed() 内部实现不同，所以用循环兜底。
    """
    for _ in range(max_steps):
        if sess.ended:
            return
        ai_res = sess.ai_move_if_needed()
        if not ai_res:
            return
        _emit_state(sess)
        if sess.ended:
            return

def _bg_ai_play(match_id: str):
    """后台跑 AI，避免阻塞 make_move/start_game 的返回，从而阻塞 sync_state 发送。"""
    sess = SESSIONS.get(match_id)
    if not sess or sess.ended:
        return
    _ai_play_until_human(sess)
    if sess.ended:
        _finalize(sess)


def _resign_winner(sess, me):
    """
    你主动离开/认输时计算 winner
    - pvp: 你离开 => 对手赢
    - pve: human 离开 => AI 赢
    """
    if sess.mode == "pvp_net":
        if me == sess.p1:
            return 2
        if me == sess.p2:
            return 1
        return -1

    # pve human = sess.p1
    if sess.game == "gomoku":
        human_side = int(getattr(sess, "human_side", 1))
        ai_side = 2 if human_side == 1 else 1
        return ai_side
    else:
        human_player = int(getattr(sess, "human_player", 1))
        ai_player = 2 if human_player == 1 else 1
        return ai_player


# -------------------------
# Connection / Auth
# -------------------------


@socketio.on("connect")
def on_connect(auth=None):
    token = None
    if isinstance(auth, dict):
        token = auth.get("token")

    if token:
        try:
            username = verify_token(token)
        except Exception as e:
            print("[connect] verify_token error:", repr(e))
            username = None

        if username:
            sid = request.sid
            old_sid = mm.bind(username, sid)
            if old_sid and old_sid != sid:
                socketio.emit("force_logout", {"msg": "This account signed in somewhere else."}, to=old_sid)
            emit("auth_ok", {"token": token, "user": {"username": username}})
            _emit_social_state(username)
            _broadcast_presence(username)

    emit("server_ready", {"ok": True})

@socketio.on("disconnect")
def on_disconnect():
    sid = _sid()
    # 这里只做解绑；是否断线判负看你需求（如果要判负，改这里：找到 match => winner=对手 => _finalize）
    username = mm.unbind_sid(sid)
    if username:
        _broadcast_presence(username)


@socketio.on("auth_register")
def on_auth_register(data):
    r = register_user((data or {}).get("username"), (data or {}).get("password"))
    if not r["ok"]:
        emit("auth_error", {"msg": r["msg"]})
        return

    sid = _sid()
    old_sid = mm.bind(r["user"]["username"], sid)
    if old_sid and old_sid != sid:
        socketio.emit("force_logout", {"msg": "This account signed in somewhere else."}, to=old_sid)

    emit("auth_ok", {"token": r["token"], "user": r["user"]})
    _emit_social_state(r["user"]["username"])
    _broadcast_presence(r["user"]["username"])


@socketio.on("auth_login")
def on_auth_login(data):
    r = login_user((data or {}).get("username"), (data or {}).get("password"))
    if not r["ok"]:
        emit("auth_error", {"msg": r["msg"]})
        return

    sid = _sid()
    old_sid = mm.bind(r["user"]["username"], sid)
    if old_sid and old_sid != sid:
        socketio.emit("force_logout", {"msg": "This account signed in somewhere else."}, to=old_sid)

    emit("auth_ok", {"token": r["token"], "user": r["user"]})
    _emit_social_state(r["user"]["username"])
    _broadcast_presence(r["user"]["username"])


@socketio.on("profile_get")
def on_profile_get(data):
    username = ((data or {}).get("username") or "").strip()
    r = get_profile(username)
    if not r["ok"]:
        emit("profile_error", {"msg": r["msg"]})
        return
    me = _me()
    if me:
        r["online"] = bool(mm.is_online(username))
        r["relation"] = _relation_between(me, username)
    emit("profile_data", _jsonable(r))


# -------------------------
# Friends / chat
# -------------------------
@socketio.on("friends_get")
def on_friends_get():
    me = _need_login()
    if not me:
        return
    emit("friends_state", _jsonable(_social_state(me)))


@socketio.on("user_search")
def on_user_search(data):
    me = _need_login()
    if not me:
        return

    target = _clean_username((data or {}).get("username"))
    if not target:
        emit("user_search_result", {"ok": False, "msg": "Please enter a username."})
        return

    doc = _get_user(target, {"username": 1})
    if not doc:
        emit("user_search_result", {"ok": False, "msg": "User not found."})
        return

    emit("user_search_result", {
        "ok": True,
        "user": {"username": doc["username"]},
        "online": bool(mm.is_online(doc["username"])),
        "relation": _relation_between(me, doc["username"]),
    })


@socketio.on("friend_request_send")
def on_friend_request_send(data):
    me = _need_login()
    if not me:
        return

    target = _clean_username((data or {}).get("username"))
    if not target:
        emit("friend_error", {"msg": "Please enter a username."})
        return
    if target == me:
        emit("friend_error", {"msg": "You cannot add yourself."})
        return
    if not _user_exists(target):
        emit("friend_error", {"msg": "User not found."})
        return

    relation = _relation_between(me, target)
    if relation == "friends":
        emit("friend_error", {"msg": "You are already friends."})
        return
    if relation == "outgoing":
        emit("friend_error", {"msg": "Friend request already sent."})
        return

    if relation == "incoming":
        users_col().update_one({"username": me}, {
            "$addToSet": {"friends": target},
            "$pull": {"friend_requests_in": target},
        })
        users_col().update_one({"username": target}, {
            "$addToSet": {"friends": me},
            "$pull": {"friend_requests_out": me},
        })
        emit("friend_request_result", {"ok": True, "accepted": True, "username": target})
        _emit_social_pair(me, target)
        return

    users_col().update_one({"username": me}, {"$addToSet": {"friend_requests_out": target}})
    users_col().update_one({"username": target}, {"$addToSet": {"friend_requests_in": me}})
    emit("friend_request_sent", {"ok": True, "to": target})
    tsid = mm.sid_of(target)
    if tsid:
        socketio.emit("friend_request_new", {"from": me}, to=tsid)
    _emit_social_pair(me, target)


@socketio.on("friend_request_reply")
def on_friend_request_reply(data):
    me = _need_login()
    if not me:
        return

    source = _clean_username((data or {}).get("username"))
    accept = bool((data or {}).get("accept"))
    if not source:
        emit("friend_error", {"msg": "Missing request sender."})
        return

    doc = _get_user(me, {"friend_requests_in": 1})
    if not doc or source not in (doc.get("friend_requests_in") or []):
        emit("friend_error", {"msg": "Friend request not found."})
        return

    if accept:
        users_col().update_one({"username": me}, {
            "$addToSet": {"friends": source},
            "$pull": {"friend_requests_in": source},
        })
        users_col().update_one({"username": source}, {
            "$addToSet": {"friends": me},
            "$pull": {"friend_requests_out": me},
        })
    else:
        users_col().update_one({"username": me}, {"$pull": {"friend_requests_in": source}})
        users_col().update_one({"username": source}, {"$pull": {"friend_requests_out": me}})

    emit("friend_request_result", {"ok": True, "accepted": accept, "username": source})
    ssid = mm.sid_of(source)
    if ssid:
        socketio.emit("friend_request_result", {"ok": True, "accepted": accept, "username": me}, to=ssid)
    _emit_social_pair(me, source)


@socketio.on("friend_remove")
def on_friend_remove(data):
    me = _need_login()
    if not me:
        return

    target = _clean_username((data or {}).get("username"))
    if not target or target == me:
        emit("friend_error", {"msg": "Invalid user."})
        return

    users_col().update_one({"username": me}, {
        "$pull": {
            "friends": target,
            "friend_requests_in": target,
            "friend_requests_out": target,
        }
    })
    users_col().update_one({"username": target}, {
        "$pull": {
            "friends": me,
            "friend_requests_in": me,
            "friend_requests_out": me,
        }
    })
    emit("friend_removed", {"ok": True, "username": target})
    tsid = mm.sid_of(target)
    if tsid:
        socketio.emit("friend_removed", {"ok": True, "username": me}, to=tsid)
    _emit_social_pair(me, target)


@socketio.on("chat_open")
def on_chat_open(data):
    me = _need_login()
    if not me:
        return

    peer = _clean_username((data or {}).get("username"))
    if not peer or not _are_friends(me, peer):
        emit("chat_error", {"msg": "You can only chat with friends."})
        return

    conv = _conversation_id(me, peer)
    rows = list(chats_col().find({"conversation": conv}, {"_id": 0}).sort("created_at", -1).limit(80))
    rows.reverse()
    emit("chat_history", {"ok": True, "peer": peer, "messages": _jsonable(rows)})


@socketio.on("chat_send")
def on_chat_send(data):
    me = _need_login()
    if not me:
        return

    peer = _clean_username((data or {}).get("username"))
    text = ((data or {}).get("text") or "").strip()
    if not peer or not _are_friends(me, peer):
        emit("chat_error", {"msg": "You can only chat with friends."})
        return
    if not text:
        emit("chat_error", {"msg": "Message cannot be empty."})
        return
    if len(text) > 500:
        emit("chat_error", {"msg": "Message is too long."})
        return

    msg = {
        "conversation": _conversation_id(me, peer),
        "participants": sorted([me, peer], key=str.lower),
        "from": me,
        "to": peer,
        "text": text,
        "created_at": _now_utc(),
    }
    chats_col().insert_one(msg)
    payload = _jsonable({k: v for k, v in msg.items() if k != "conversation"})
    emit("chat_message", {"peer": peer, "message": payload})
    psid = mm.sid_of(peer)
    if psid:
        socketio.emit("chat_message", {"peer": me, "message": payload}, to=psid)


# -------------------------
# PvP: username challenge
# -------------------------
@socketio.on("challenge_user")
def on_challenge_user(data):
    me = _need_login()
    if not me:
        return

    sid = _sid()
    if mm.in_match(sid):
        emit("error", {"msg": "You are already in a match."})
        return

    target = ((data or {}).get("targetUsername") or "").strip()
    game = (data or {}).get("game")

    if not target:
        emit("error", {"msg": "Opponent username is empty."})
        return
    if target == me:
        emit("error", {"msg": "You cannot challenge yourself."})
        return
    if game not in SUPPORTED_GAMES:
        emit("error", {"msg": "Unsupported game."})
        return

    tsid = mm.sid_of(target)
    if not tsid:
        emit("error", {"msg": "Opponent is offline."})
        return
    if mm.in_match(tsid):
        emit("error", {"msg": "Opponent is already in a match."})
        return

    cid = str(uuid.uuid4())
    CHALLENGES[cid] = {"from": me, "to": target, "game": game}

    socketio.emit("challenge_request", {"challengeId": cid, "from": me, "game": game}, to=tsid)
    emit("challenge_sent", {"challengeId": cid, "to": target, "game": game})


@socketio.on("challenge_reply")
def on_challenge_reply(data):
    me = _need_login()
    if not me:
        return

    sid = _sid()
    cid = (data or {}).get("challengeId")
    accept = bool((data or {}).get("accept"))

    ch = CHALLENGES.get(cid)
    if not ch:
        emit("error", {"msg": "Challenge expired."})
        return
    if ch["to"] != me:
        emit("error", {"msg": "This challenge was not sent to you."})
        return

    from_user = ch["from"]
    to_user = ch["to"]
    game = ch["game"]

    fsid = mm.sid_of(from_user)
    if not fsid:
        emit("error", {"msg": "Challenger is offline."})
        CHALLENGES.pop(cid, None)
        return

    if not accept:
        socketio.emit("challenge_declined", {"challengeId": cid, "by": me}, to=fsid)
        CHALLENGES.pop(cid, None)
        return

    # 双方必须都不在对局中
    if mm.in_match(fsid) or mm.in_match(sid):
        emit("error", {"msg": "One player is already in a match."})
        CHALLENGES.pop(cid, None)
        return

    # seat random
    if random.random() < 0.5:
        p1, p2 = from_user, to_user
    else:
        p1, p2 = to_user, from_user

    if game == "gomoku":
        sess = GomokuSession(
            mode="pvp_net",
            p1=p1,
            p2=p2,
            size=GOMOKU_BOARD_SIZE,
            n_in_row=GOMOKU_N_IN_ROW,
        )
    else:
        sess = CheckersSession(mode="pvp_net", p1=p1, p2=p2)

    sess.sids = [fsid, sid]
    sess.room = sess.match_id
    SESSIONS[sess.match_id] = sess

    join_room(sess.room, sid=fsid)
    join_room(sess.room, sid=sid)
    mm.set_in_match(fsid, sess.match_id)
    mm.set_in_match(sid, sess.match_id)

    p1_sid = mm.sid_of(p1)
    p2_sid = mm.sid_of(p2)

    socketio.emit("match_found", {"matchId": sess.match_id, "game": game, "mode": "pvp_net", "yourPlayer": 1, "opponent": p2}, to=p1_sid)
    socketio.emit("match_found", {"matchId": sess.match_id, "game": game, "mode": "pvp_net", "yourPlayer": 2, "opponent": p1}, to=p2_sid)

    _emit_state(sess)
    CHALLENGES.pop(cid, None)


# -------------------------
# PvE: start_game
# -------------------------
@socketio.on("start_game")
def on_start_game(data):
    me = _need_login()
    if not me:
        return

    sid = _sid()
    game = (data or {}).get("game")
    mode = (data or {}).get("mode")
    difficulty = _normalize_difficulty((data or {}).get("difficulty"))

    if mode not in ("pve", "local"):
        emit("error", {"msg": "Only pve / local are supported."})
        return
    if mm.in_match(sid):
        emit("error", {"msg": "You are already in a match."})
        return
    if game not in SUPPORTED_GAMES:
        emit("error", {"msg": "Unsupported game."})
        return

    if game == "gomoku":
        
        if mode == "pve":
            human_side = int(data.get("humanSide", 1))  # 默认=1 人先手 ✅
            pool, ai_cfg = _get_gomoku_pool(difficulty)
            ai_agent = GomokuAZAgent(
                pool,
                c_puct=ai_cfg["c_puct"],
                n_playout=ai_cfg["playouts"],
                mcts_batch_size=GOMOKU_MCTS_BATCH_SIZE,
                tactical_mode=ai_cfg["tactics"],
            )
            sess = GomokuSession(
                mode="pve",
                p1=me,
                p2="AI",
                size=GOMOKU_BOARD_SIZE,
                n_in_row=GOMOKU_N_IN_ROW,
                human_side=human_side,
                ai_agent=ai_agent,
            )
            sess.ai_difficulty = difficulty
            print("[GOMOKU] create session ai_agent = {}, difficulty = {}, playouts = {}, tactics = {}".format(
                type(ai_agent).__name__ if ai_agent else None,
                difficulty,
                ai_cfg["playouts"],
                ai_cfg["tactics"],
            ))
            your_player = human_side
        else:  # local
            sess = GomokuSession(
                mode="local",
                p1=me,
                p2=me,
                size=GOMOKU_BOARD_SIZE,
                n_in_row=GOMOKU_N_IN_ROW,
                human_side=1,
                ai_agent=None,
            )
            your_player = 1
       
    else:  # checkers
        if mode == "pve":
            human_player = int((data or {}).get("humanPlayer", 1))
           
            pool, ai_cfg = _get_checkers_pool(difficulty)
            ai_agent = CheckersAZAgent(
                pool,
                c_puct=ai_cfg["c_puct"],
                n_playout=ai_cfg["playouts"],
                mcts_batch_size=CHECKERS_MCTS_BATCH_SIZE,
                safety_gap=CHECKERS_SAFETY_GAP,
                tactical_mode=ai_cfg["tactics"],
            )
            sess = CheckersSession(mode="pve", p1=me, p2="AI", human_player=human_player, ai_agent=ai_agent)
            sess.ai_difficulty = difficulty
            print("[CHECKER] create session ai_agent = {}, difficulty = {}, playouts = {}, tactics = {}".format(
                type(ai_agent).__name__ if ai_agent else None,
                difficulty,
                ai_cfg["playouts"],
                ai_cfg["tactics"],
            ))
            your_player = human_player
        else:  # local
            sess = CheckersSession(mode="local", p1=me, p2=me, human_player=1)
            your_player = 1

    sess.sids = [sid]
    sess.room = sess.match_id
    SESSIONS[sess.match_id] = sess

    join_room(sess.room, sid=sid)
    mm.set_in_match(sid, sess.match_id)

    ai_label = AI_DIFFICULTY_LABELS.get(difficulty, "Normal")
    opponent_label = "AI ({})".format(ai_label) if sess.mode == "pve" else "AI"
    emit("match_found", {
        "matchId": sess.match_id,
        "game": game,
        "mode": sess.mode,
        "yourPlayer": your_player,
        "opponent": opponent_label,
        "difficulty": difficulty if sess.mode == "pve" else None,
    })
    _emit_state(sess)

    # ✅ 让 start_game 立刻返回，先把棋盘发到前端；AI 放后台跑
    if sess.mode == "pve":
        socketio.start_background_task(_bg_ai_play, sess.match_id)


# -------------------------
# Gameplay
# -------------------------
@socketio.on("make_move")
def on_make_move(data):
    me = _need_login()
    if not me:
        return

    sid = _sid()
    match_id = _get_match_id(data)
    if not match_id:
        emit("error", {"msg": "Missing matchId."})
        return

    sess = _get_session_or_err(match_id)
    if not sess:
        return

    yp = _your_player(sess, me)
    if yp is None:
        emit("error", {"msg": "You are not a player in this match."})
        return

    # -------- gomoku --------
    if sess.game == "gomoku":
        idx = int((data or {}).get("index", -1))
        res = sess.make_move(idx, yp)
        if not res["ok"]:
            emit("invalid_move", {"msg": res["msg"]})
            return

        _emit_state(sess)
        if sess.ended:
            _finalize(sess)
            return

        if sess.mode == "pve":
            _ai_play_until_human(sess)
            if sess.ended:
                _finalize(sess)
        return

    # -------- checkers --------
    frm = int((data or {}).get("from", -1))
    to = int((data or {}).get("to", -1))
    res = sess.make_move(frm, to, yp)
    if not res["ok"]:
        emit("invalid_move", {"msg": res["msg"]})
        return

    _emit_state(sess)
    if sess.ended:
        _finalize(sess)
        return

    
    if sess.mode == "pve":
    # ✅ 只在“轮到 AI”时才启动后台（强制连吃导致仍是你走，就不会启动）
            ai_player = 2 if int(getattr(sess, "human_player", 1)) == 1 else 1
            if int(getattr(sess, "current_player", 1)) == ai_player:
                socketio.start_background_task(_bg_ai_play, sess.match_id)
    return

@socketio.on("request_undo")
def on_request_undo(data):
    me = _need_login()
    if not me:
        return

    sid = _sid()
    match_id = _get_match_id(data)
    if not match_id:
        emit("error", {"msg": "Missing matchId."})
        return

    sess = _get_session_or_err(match_id)
    if not sess:
        return

    # 必须是本局玩家
    if sid not in (sess.sids or []):
        emit("error", {"msg": "You are not in this match."})
        return

    # Online PvP 一般不允许悔棋（否则会扯皮）
    if sess.mode == "pvp_net":
        emit("invalid_move", {"msg": "Online PvP does not support undo."})
        return

    # 规则：Gomoku 的 PvE 默认撤两手（你 + AI），撤完还是你走
    steps = 1
    if sess.game == "gomoku" and sess.mode == "pve":
        steps = 2

    if not hasattr(sess, "undo"):
        emit("invalid_move", {"msg": "Undo is not supported for this game yet."})
        return

    r = sess.undo(steps=steps)
    if not r.get("ok"):
        emit("invalid_move", {"msg": r.get("msg", "Undo failed.")})
        return

    # 广播新局面
    _emit_state(sess)

    # 若撤回后变成 AI 回合（保险起见），让 AI 自动走到轮到 human
    _ai_play_until_human(sess)
    if sess.ended:
        _finalize(sess)


@socketio.on("leave_match")
def on_leave_match(data):
    me = _need_login()
    if not me:
        return

    sid = _sid()
    match_id = _get_match_id(data)
    if not match_id:
        return

    sess = SESSIONS.get(match_id)
    
    if not sess:
        mm.clear_match(sid)
        try:
            leave_room(match_id, sid=sid)
        except Exception:
            pass
        return
    if sess.mode == "local" and not sess.ended:
        sess.winner = -1
        sess.ended = True
        _finalize(sess)
        return
    # 对局未结束：离开视为认输（pvp/pve 都一样合理）
    if not sess.ended:
        # ✅ 先把离开的人踢出房间，避免他收到 game_over 弹窗
        _detach_sid(sess, sid)

        # ✅ local 也按“认输”处理：当前走子方判负（另一方胜）
        if sess.mode == "local":
            cp = int(getattr(sess, "current_player", 1))
            sess.winner = 2 if cp == 1 else 1
        else:
            # pve/pvp_net：离开者判负
            sess.winner = _resign_winner(sess, me)

        sess.ended = True
        _finalize(sess)
        return

    # 已结束：只清理自己
    mm.clear_match(sid)
    try:
        leave_room(match_id, sid=sid)
    except Exception:
        pass


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    server_host = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "5000"))
    public_mode = os.getenv("PUBLIC_MODE", "0") == "1"
    if public_mode and JWT_SECRET == "dev-secret-change-me":
        raise RuntimeError("PUBLIC_MODE=1 requires setting a strong JWT_SECRET environment variable.")

    socketio.run(
        app,
        host=server_host,
        port=server_port,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
