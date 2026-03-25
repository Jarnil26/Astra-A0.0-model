"""
config.py — centralised environment variable loading for Astra A0.0 Backend
Set these on Render dashboard (or in .env for local dev).
"""
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env if present

# ── MongoDB ────────────────────────────────────────────────────────────────────
# User accounts (register / login / reset password)
USER_MONGO_URI: str = os.environ.get(
    "USER_MONGO_URI",
    "mongodb+srv://risusolutions_astra_user:uP8MxhoVmVeBpds9@cluster0.sceqrkf.mongodb.net/?appName=Cluster0"
)
USER_DB_NAME: str = "astra_users"

# Chat session history
CHAT_MONGO_URI: str = os.environ.get(
    "CHAT_MONGO_URI",
    "mongodb+srv://risusolutions_db_user:94fT2hhqgWhCUODr@cluster0.h2wquwv.mongodb.net/?appName=Cluster0"
)
CHAT_DB_NAME: str = "astra_clinical"

# ── JWT ────────────────────────────────────────────────────────────────────────
JWT_SECRET: str = os.environ.get("JWT_SECRET", "change-me-in-production-use-64-char-random-string")
JWT_ALGORITHM: str = "HS256"
JWT_EXPIRE_DAYS: int = int(os.environ.get("JWT_EXPIRE_DAYS", "7"))

# ── Email / SMTP ───────────────────────────────────────────────────────────────
SMTP_HOST: str = os.environ.get("SMTP_HOST", "smtp.zoho.in")
SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER: str = os.environ.get("SMTP_USER", "connect@byteastra.in")
SMTP_PASS: str = os.environ.get("SMTP_PASS", "")          # set on Render
EMAIL_FROM_NAME: str = "Astra A0.0 Clinical Engine"
EMAIL_FROM: str = f"{EMAIL_FROM_NAME} <{SMTP_USER}>"

# ── App ────────────────────────────────────────────────────────────────────────
APP_URL: str = os.environ.get("APP_URL", "http://localhost:8000")
# Token expiry (minutes)
EMAIL_TOKEN_EXPIRE_MINUTES: int = 60       # email verification
RESET_TOKEN_EXPIRE_MINUTES: int = 30       # password reset

# ── Clinical engine ────────────────────────────────────────────────────────────
FAISS_INDEX_PATH: str = os.environ.get("FAISS_INDEX_PATH", "data/ayurveda.index")
DB_PATH: str = os.environ.get("DB_PATH", "data/ayurveda_ai.db")
PREVALENCE_PATH: str = os.environ.get("PREVALENCE_PATH", "data/disease_prevalence.json")
