# -*- coding: utf-8 -*-
"""
auth/email_service.py — sends transactional emails via SMTP
Sender: connect@byteastra.in
"""
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger("astra.email")


def _send(to_email: str, subject: str, html_body: str):
    """Low-level SMTP send. Raises on failure."""
    from config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_FROM

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [to_email], msg.as_string())
        logger.info("Email sent to %s | Subject: %s", to_email, subject)
    except Exception as e:
        logger.error("Email failed to %s: %s", to_email, e)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Templates
# ─────────────────────────────────────────────────────────────────────────────

_BASE_STYLE = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9; margin: 0; padding: 20px; }
  .card { background: #ffffff; border-radius: 12px; max-width: 520px; margin: auto;
          padding: 40px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
  .logo { font-size: 22px; font-weight: 700; color: #4F46E5; margin-bottom: 8px; }
  .sub  { color: #6B7280; font-size: 13px; margin-bottom: 28px; }
  h2    { color: #111827; font-size: 20px; margin: 0 0 12px; }
  p     { color: #374151; font-size: 15px; line-height: 1.6; margin: 8px 0; }
  .btn  { display: inline-block; margin: 24px 0; padding: 13px 30px;
          background: linear-gradient(135deg, #4F46E5, #7C3AED);
          color: #ffffff !important; text-decoration: none;
          border-radius: 8px; font-weight: 600; font-size: 15px; }
  .note { font-size: 12px; color: #9CA3AF; margin-top: 24px; border-top: 1px solid #E5E7EB; padding-top: 16px; }
</style>
"""


def send_verification_email(to_email: str, name: str, token: str):
    from config import APP_URL
    verify_url = f"{APP_URL}/auth/verify-email?token={token}"
    html = f"""<!DOCTYPE html><html><head>{_BASE_STYLE}</head><body>
    <div class="card">
      <div class="logo">⚕️ Astra A0.0</div>
      <div class="sub">Clinical Intelligence Engine</div>
      <h2>Welcome, {name}! Verify your email</h2>
      <p>Thank you for registering. Click the button below to verify your email address and activate your account.</p>
      <a href="{verify_url}" class="btn">✅ Verify Email Address</a>
      <p>Or copy this link:<br><small>{verify_url}</small></p>
      <div class="note">This link expires in 60 minutes. If you didn't register, ignore this email.</div>
    </div></body></html>"""
    _send(to_email, "Verify your Astra A0.0 account", html)


def send_reset_email(to_email: str, name: str, token: str):
    from config import APP_URL
    reset_url = f"{APP_URL}/auth/reset-password?token={token}"
    html = f"""<!DOCTYPE html><html><head>{_BASE_STYLE}</head><body>
    <div class="card">
      <div class="logo">⚕️ Astra A0.0</div>
      <div class="sub">Clinical Intelligence Engine</div>
      <h2>Reset your password</h2>
      <p>Hi {name}, we received a request to reset your Astra A0.0 password.</p>
      <a href="{reset_url}" class="btn">🔑 Reset Password</a>
      <p>Or copy this link:<br><small>{reset_url}</small></p>
      <div class="note">This link expires in 30 minutes. If you didn't request a reset, you can safely ignore this email.</div>
    </div></body></html>"""
    _send(to_email, "Reset your Astra A0.0 password", html)
