"""
api.py — Astra A0.0 Clinical Engine Backend
Main FastAPI application entry point.

Routers:
  /auth/*   → auth.routes  (register, verify, login, forgot, reset, me)
  /chat/*   → chat.routes  (new, message, sessions, history, delete, rename)
  /health   → liveness check
  /docs     → Swagger UI (auto-generated)
"""
import logging
import time
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("astra.api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Astra A0.0 Clinical Engine API",
    description=(
        "🩺 **Astra A0.0** — Multilingual Ayurvedic diagnostic engine.\n\n"
        "- Accepts symptoms in **English, Hindi, Gujarati, Hinglish**\n"
        "- ChatGPT-style session memory (symptoms accumulate across turns)\n"
        "- Full user auth with email verification\n\n"
        "**Auth flow:** Register → Verify Email → Login → use Bearer token\n\n"
        "**Chat flow:** POST /chat/new → POST /chat/{id}/message (repeat) → GET /chat/sessions"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow all origins for frontend access ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from auth.routes import router as auth_router
from chat.routes import router as chat_router

app.include_router(auth_router)
app.include_router(chat_router)

# ── Engine startup ────────────────────────────────────────────────────────────
_startup_error: str = ""
_engine_ready: bool = False


@app.on_event("startup")
def startup():
    global _engine_ready, _startup_error
    t0 = time.time()
    logger.info("=== Starting Astra A0.0 Backend ===")

    # Pre-warm DB connections
    try:
        from db.user_db import get_users_collection
        get_users_collection()
    except Exception as e:
        _startup_error = f"User DB: {e}"
        logger.error("User DB startup failed: %s", e)

    try:
        from db.chat_db import get_sessions_collection
        get_sessions_collection()
    except Exception as e:
        _startup_error += f" | Chat DB: {e}"
        logger.error("Chat DB startup failed: %s", e)

    # Pre-warm clinical engine
    try:
        from clinical.engine import ClinicalEngine
        ClinicalEngine.instance()
        _engine_ready = True
    except FileNotFoundError as e:
        _startup_error += f" | Engine: {e}"
        logger.warning("Clinical engine not ready (run build_astra.py): %s", e)
    except Exception as e:
        _startup_error += f" | Engine: {e}"
        logger.error("Clinical engine startup failed: %s", e)

    logger.info("Startup complete in %.2fs | engine_ready=%s", time.time() - t0, _engine_ready)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness probe — Render uses /health to verify the service is up."""
    return {
        "status": "ok",
        "engine": "Astra A0.0 v2",
        "engine_ready": _engine_ready,
        "startup_error": _startup_error or None,
    }


@app.get("/", tags=["System"], include_in_schema=False)
def root():
    return {
        "message": "Astra A0.0 Clinical Engine API",
        "docs": "/docs",
        "health": "/health",
    }
