"""
auth/models.py — Pydantic request/response models for auth endpoints
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=80, example="Risu")
    email: EmailStr = Field(..., example="risu@example.com")
    password: str = Field(..., min_length=6, example="StrongPass123")


class VerifyEmailRequest(BaseModel):
    token: str = Field(..., example="uuid-token-from-email")


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., example="risu@example.com")
    password: str = Field(..., example="StrongPass123")


class ForgotPasswordRequest(BaseModel):
    email: EmailStr = Field(..., example="risu@example.com")


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., example="uuid-token-from-email")
    new_password: str = Field(..., min_length=6, example="NewPass456")


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    is_verified: bool


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
