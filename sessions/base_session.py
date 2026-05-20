# -*- coding: utf-8 -*-
import uuid
from datetime import datetime, timezone

def _now():
    return datetime.now(timezone.utc)

class BaseSession:
    def __init__(self, game: str, mode: str):
        self.match_id = str(uuid.uuid4())
        self.game = game
        self.mode = mode
        self.created_at = _now()
        self.winner = None  # 1/2/-1
        self.ended = False

    def summary(self):
        return {
            "matchId": self.match_id,
            "game": self.game,
            "mode": self.mode,
            "createdAt": self.created_at.isoformat()
        }