"""
db/user_db.py — MongoDB connection for user accounts (auth)
Cluster: cluster0.sceqrkf.mongodb.net
"""
import logging
from typing import Optional
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure

logger = logging.getLogger("astra.db.users")

_client: Optional[MongoClient] = None
_users_col: Optional[Collection] = None


def get_users_collection() -> Collection:
    """Return the users collection, initialising the connection if needed."""
    global _client, _users_col
    if _users_col is not None:
        return _users_col

    from config import USER_MONGO_URI, USER_DB_NAME
    try:
        _client = MongoClient(USER_MONGO_URI, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        db = _client[USER_DB_NAME]
        _users_col = db["users"]

        # Indexes
        _users_col.create_index([("email", ASCENDING)], unique=True, background=True)
        _users_col.create_index([("verification_token", ASCENDING)], background=True, sparse=True)
        _users_col.create_index([("reset_token", ASCENDING)], background=True, sparse=True)

        logger.info("User DB connected: %s.users", USER_DB_NAME)
    except ConnectionFailure as e:
        logger.error("User DB connection failed: %s", e)
        raise

    return _users_col
