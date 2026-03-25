"""
auth/service.py — password hashing, JWT creation/verification, user CRUD
"""
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from passlib.context import CryptContext
from jose import jwt, JWTError

logger = logging.getLogger("astra.auth")

# Password hashing
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(user_id: str, email: str) -> str:
    from config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_DAYS
    now = time.time()
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + JWT_EXPIRE_DAYS * 86400,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Returns payload dict or raises JWTError."""
    from config import JWT_SECRET, JWT_ALGORITHM
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


# ── User CRUD ─────────────────────────────────────────────────────────────────

def _col():
    from db.user_db import get_users_collection
    return get_users_collection()


def create_user(name: str, email: str, password: str) -> Dict[str, Any]:
    """
    Insert a new user document. Returns the created doc.
    Raises ValueError if email already registered.
    """
    existing = _col().find_one({"email": email.lower()})
    if existing:
        raise ValueError("Email already registered.")

    token = str(uuid.uuid4())
    from config import EMAIL_TOKEN_EXPIRE_MINUTES
    now = time.time()
    doc = {
        "name": name.strip(),
        "email": email.lower().strip(),
        "password_hash": hash_password(password),
        "is_verified": False,
        "verification_token": token,
        "verification_expires": now + EMAIL_TOKEN_EXPIRE_MINUTES * 60,
        "reset_token": None,
        "reset_expires": None,
        "created_at": now,
        "last_login": None,
    }
    result = _col().insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    return _col().find_one({"email": email.lower().strip()})


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    from bson import ObjectId
    return _col().find_one({"_id": ObjectId(user_id)})


def verify_email_token(token: str) -> bool:
    """Verify email verification token. Returns True if successful."""
    now = time.time()
    user = _col().find_one({
        "verification_token": token,
        "is_verified": False,
        "verification_expires": {"$gt": now},
    })
    if not user:
        return False
    _col().update_one(
        {"_id": user["_id"]},
        {"$set": {"is_verified": True, "verification_token": None, "verification_expires": None}}
    )
    return True


def create_reset_token(email: str) -> Optional[str]:
    """
    Generate password reset token for user. Returns token or None if user not found.
    """
    user = get_user_by_email(email)
    if not user:
        return None
    token = str(uuid.uuid4())
    from config import RESET_TOKEN_EXPIRE_MINUTES
    now = time.time()
    _col().update_one(
        {"_id": user["_id"]},
        {"$set": {"reset_token": token, "reset_expires": now + RESET_TOKEN_EXPIRE_MINUTES * 60}}
    )
    return token


def reset_password(token: str, new_password: str) -> bool:
    """Reset password using a valid reset token. Returns True if successful."""
    now = time.time()
    user = _col().find_one({
        "reset_token": token,
        "reset_expires": {"$gt": now},
    })
    if not user:
        return False
    _col().update_one(
        {"_id": user["_id"]},
        {"$set": {
            "password_hash": hash_password(new_password),
            "reset_token": None,
            "reset_expires": None,
        }}
    )
    return True


def update_last_login(user_id) -> None:
    from bson import ObjectId
    _col().update_one({"_id": ObjectId(str(user_id))}, {"$set": {"last_login": time.time()}})
