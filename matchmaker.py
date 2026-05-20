# -*- coding: utf-8 -*-

class Matchmaker:
    def __init__(self):
        # username -> socket sid
        self.user_sid = {}
        # sid -> username
        self.sid_user = {}
        # sid -> current matchId（防止同时多开对局）
        self.sid_match = {}

    def bind(self, username: str, sid: str):
        # 单账号单在线：新登录顶掉旧的
        old_sid = self.user_sid.get(username)
        self.user_sid[username] = sid
        self.sid_user[sid] = username
        return old_sid

    def unbind_sid(self, sid: str):
        u = self.sid_user.pop(sid, None)
        if u and self.user_sid.get(u) == sid:
            self.user_sid.pop(u, None)
        self.sid_match.pop(sid, None)
        return u

    def username_of(self, sid: str):
        return self.sid_user.get(sid)

    def sid_of(self, username: str):
        return self.user_sid.get(username)

    def is_online(self, username: str):
        return username in self.user_sid

    def set_in_match(self, sid: str, match_id: str):
        self.sid_match[sid] = match_id

    def clear_match(self, sid: str):
        self.sid_match.pop(sid, None)

    def in_match(self, sid: str):
        return sid in self.sid_match