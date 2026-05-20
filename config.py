# -*- coding: utf-8 -*-
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://xiarenee17_db_user:lxx123@cluster0.4i36bjl.mongodb.net/")
DB_NAME = os.getenv("DB_NAME", "zeromind_db")

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "7"))

SOCKET_CORS = os.getenv("SOCKET_CORS", "*")
