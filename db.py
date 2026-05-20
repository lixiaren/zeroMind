# -*- coding: utf-8 -*-
from pymongo import MongoClient, ASCENDING
from config import MONGO_URI, DB_NAME

_client = None

def get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[DB_NAME]

def users_col():
    return get_db()["users"]

def chats_col():
    return get_db()["chats"]

def init_indexes():
    # username 唯一
    users_col().create_index([("username", ASCENDING)], unique=True)
    chats_col().create_index([("conversation", ASCENDING), ("created_at", ASCENDING)])
    chats_col().create_index([("participants", ASCENDING), ("created_at", ASCENDING)])
