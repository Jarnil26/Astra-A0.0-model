"""
auth/routes.py — /auth/* endpoints
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from auth.models import (
    RegisterRequest, VerifyEmailRequest, LoginRequest,
    ForgotPasswordRequest, ResetPasswordRequest,
    TokenResponse, UserResponse,
)
import auth.service as svc

logger = logging.getLogger("astra.auth.routes")
router = APIRouter(prefix="/auth", tags=["Auth"])
_bearer = HTTPBearer()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _user_response(doc) -> UserResponse:
    return UserResponse(
        id=str(doc["_id"]),
        name=doc["name"],
        email=doc["email"],
        is_verified=doc.get("is_verified", False),
    )


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    """Dependency — validates Bearer JWT and returns user doc."""
    from jose import JWTError
    try:
        payload = svc.decode_access_token(credentials.credentials)
        user = svc.get_user_by_id(payload["sub"])
        if not user:
            raise HTTPException(status_code=401, detail="User not found.")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", summary="Register a new account")
def register(req: RegisterRequest):
    """
    Creates a new user and sends a verification email.
    User must verify email before logging in.
    """
    try:
        user = svc.create_user(req.name, req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Send verification email (non-blocking — log errors, don't fail request)
    try:
        from auth.email_service import send_verification_email
        send_verification_email(user["email"], user["name"], user["verification_token"])
    except Exception as e:
        logger.warning("Could not send verification email to %s: %s", user["email"], e)

    return {
        "message": "Registration successful. Please check your email to verify your account.",
        "email": user["email"],
    }


@router.post("/verify-email", summary="Verify email with token from email link")
def verify_email(req: VerifyEmailRequest):
    """Activates account. Token is sent in the verification email."""
    success = svc.verify_email_token(req.token)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid or expired verification token.")
    return {"message": "Email verified successfully. You can now log in."}


@router.get("/verify-email", summary="Verify email via GET (for email link clicks)", include_in_schema=False)
def verify_email_get(token: str):
    """Handles clicking the link in the email directly."""
    success = svc.verify_email_token(token)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid or expired verification token.")
    return {"message": "Email verified successfully. You can now log in."}


@router.post("/login", response_model=TokenResponse, summary="Login and get JWT token")
def login(req: LoginRequest):
    """Returns a JWT access token valid for 7 days."""
    user = svc.get_user_by_email(req.email)
    if not user or not svc.verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not user.get("is_verified"):
        raise HTTPException(
            status_code=403,
            detail="Email not verified. Please check your inbox and verify your email first."
        )
    token = svc.create_access_token(str(user["_id"]), user["email"])
    svc.update_last_login(user["_id"])
    return TokenResponse(access_token=token, user=_user_response(user))


@router.post("/forgot-password", summary="Send password reset email")
def forgot_password(req: ForgotPasswordRequest):
    """
    Sends a password reset link to the registered email.
    Always returns success (to prevent email enumeration attacks).
    """
    token = svc.create_reset_token(req.email)
    if token:
        user = svc.get_user_by_email(req.email)
        try:
            from auth.email_service import send_reset_email
            send_reset_email(req.email, user["name"], token)
        except Exception as e:
            logger.warning("Could not send reset email to %s: %s", req.email, e)
    return {"message": "If that email is registered, a password reset link has been sent."}


@router.post("/reset-password", summary="Reset password with token from email")
def reset_password(req: ResetPasswordRequest):
    """Resets password using the token from the reset email."""
    success = svc.reset_password(req.token, req.new_password)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token.")
    return {"message": "Password reset successfully. You can now log in."}


@router.get("/me", response_model=UserResponse, summary="Get current user profile")
def get_me(current_user=Depends(get_current_user)):
    """Returns the authenticated user's profile. Requires Bearer token."""
    return _user_response(current_user)
