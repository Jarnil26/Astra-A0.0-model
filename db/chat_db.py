"""
db/chat_db.py — MongoDB connection for chat sessions
Cluster: cluster0.h2wquwv.mongodb.net
"""
import logging
from typing import Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure

logger = logging.getLogger("astra.db.chat")

_client: Optional[MongoClient] = None
_sessions_col: Optional[Collection] = None


def get_sessions_collection() -> Collection:
    """Return the chat sessions collection, initialising connection if needed."""
    global _client, _sessions_col
    if _sessions_col is not None:
        return _sessions_col

    from config import CHAT_MONGO_URI, CHAT_DB_NAME
    try:
        _client = MongoClient(CHAT_MONGO_URI, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        db = _client[CHAT_DB_NAME]
        _sessions_col = db["chat_sessions"]

        # Indexes
        _sessions_col.create_index([("user_id", ASCENDING)], background=True)
        _sessions_col.create_index([("user_id", ASCENDING), ("last_active", DESCENDING)], background=True)

        logger.info("Chat DB connected: %s.chat_sessions", CHAT_DB_NAME)
    except ConnectionFailure as e:
        logger.error("Chat DB connection failed: %s", e)
        raise

    return _sessions_col
