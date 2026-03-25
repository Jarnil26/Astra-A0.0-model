"""
filter_layer/response_formatter.py
Formats clinical engine output into a human-readable, emoji-rich response.
Also provides safety warning detection for critical symptoms.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Critical symptoms that warrant an emergency warning
# ---------------------------------------------------------------------------
CRITICAL_SYMPTOMS = {
    "chest pain", "chest tightness", "chest pressure",
    "breathing difficulty", "shortness of breath", "difficulty breathing",
    "unconsciousness", "loss of consciousness", "fainting",
    "severe chest pain", "heart attack", "stroke",
    "seizures", "convulsions",
    "severe bleeding", "blood in stool", "blood in urine",
    "high fever", "very high fever",
    "paralysis", "sudden numbness",
}

DISCLAIMER = (
    "⚠️ *Disclaimer:* This analysis is AI-generated for informational purposes only. "
    "It is NOT a substitute for professional medical advice. "
    "Please consult a qualified healthcare professional for proper diagnosis and treatment."
)

SAFETY_WARNING = (
    "🚨 **URGENT SAFETY ALERT**\n"
    "One or more of your symptoms may indicate a serious medical condition.\n"
    "**Please seek immediate medical attention or call emergency services.**\n"
    "Do not delay — your health is the priority."
)

FALLBACK_MESSAGE = (
    "🤔 I couldn't clearly identify any specific symptoms from your input.\n"
    "Please describe your symptoms more clearly, e.g.:\n"
    "  • _'I have fever and headache'_\n"
    "  • _'mujhe bukhar aur sar dard hai'_\n"
    "  • _'mane tav aayo che'_"
)


def check_critical_symptoms(symptoms: List[str]) -> bool:
    """Return True if any critical/emergency symptom is present."""
    sym_set = {s.lower().strip() for s in symptoms}
    return bool(sym_set & CRITICAL_SYMPTOMS)


def format_clinical_response(
    result: Dict[str, Any],
    session_symptoms: List[str],
    language: str = "en"
) -> str:
    """
    Format clinical engine output using the Astra A0 report template.

    Template:
        Symptom summary → top 2 diseases → home remedies →
        herbs → yoga → lifestyle → clinical note → disclaimer
    """
    lines: List[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    lines.append("\n" + "=" * 56)
    lines.append("  ASTRA A0  |  CLINICAL DIAGNOSTIC REPORT")
    lines.append("=" * 56)

    # ── Symptoms Analyzed ─────────────────────────────────────────────────
    if session_symptoms:
        sym_str = ", ".join(s.title() for s in session_symptoms)
        lines.append(f"\nSymptoms Analyzed : {sym_str}")

    # ── Top Diagnoses ─────────────────────────────────────────────────────
    predictions = result.get("predictions", [])

    if predictions:
        top1 = predictions[0].get("disease", "Unknown")
        top2 = predictions[1].get("disease", "Unknown") if len(predictions) > 1 else None

        if top2:
            lines.append(
                f"\n👉  Based on your symptoms, it may be related to infections like "
                f"{top1} or {top2} (most common possibilities)."
            )
        else:
            lines.append(
                f"\n👉  Based on your symptoms, the most likely condition is {top1}."
            )

    else:
        lines.append("\n👉  No strong diagnosis found. Please describe more symptoms.")

    remedies = result.get("remedies", {})

    # ── Home Remedies ─────────────────────────────────────────────────────
    home_rem = remedies.get("home_remedies", [])
    lines.append("\n" + "-" * 56)
    lines.append("  Home Remedies")
    lines.append("-" * 56)
    if home_rem:
        for r in home_rem[:5]:
            lines.append(f"  - {r.title()}")
    else:
        lines.append("  - Rest well and stay hydrated")
        lines.append("  - Drink warm fluids")
        lines.append("  - Consult a healthcare provider for persistent symptoms")

    # ── Herbs ─────────────────────────────────────────────────────────────
    herbs = remedies.get("herbs", [])
    lines.append("\n" + "-" * 56)
    lines.append("  Herbs")
    lines.append("-" * 56)
    if herbs:
        for h in herbs[:4]:
            lines.append(f"  - {h.title()}")
    else:
        lines.append("  - Tulsi (Holy Basil)")
        lines.append("  - Ginger (Adrak)")
        lines.append("  - Turmeric (Haldi)")

    # ── Yoga & Rest ───────────────────────────────────────────────────────
    yoga = remedies.get("yoga", [])
    lines.append("\n" + "-" * 56)
    lines.append("  Yoga & Rest")
    lines.append("-" * 56)
    if yoga:
        for y in yoga[:3]:
            lines.append(f"  - {y.title()}")
    else:
        lines.append("  - Shavasana (Complete rest pose)")
        lines.append("  - Pranayama (Breathing exercises)")

    # ── Lifestyle Tips ────────────────────────────────────────────────────
    lifestyle = remedies.get("lifestyle", [])
    lines.append("\n" + "-" * 56)
    lines.append("  Lifestyle Tips")
    lines.append("-" * 56)
    if lifestyle:
        for lf in lifestyle[:4]:
            lines.append(f"  - {lf.title()}")
    else:
        lines.append("  - Get adequate sleep (7-8 hours)")
        lines.append("  - Drink 8-10 glasses of warm water daily")
        lines.append("  - Avoid processed and fried food")

    # ── Clinical Note ─────────────────────────────────────────────────────
    notes = result.get("notes", "")
    dosha = result.get("dosha", [])
    lines.append("\n" + "-" * 56)
    lines.append("  Clinical Note")
    lines.append("-" * 56)
    if notes:
        lines.append(f"  {notes}")
    else:
        lines.append("  Based on symptom analysis and Ayurvedic dosha mapping.")
    if dosha:
        lines.append(f"  Dosha Imbalance : {' + '.join(dosha)}")

    # ── Disclaimer ────────────────────────────────────────────────────────
    lines.append("\n" + "=" * 56)
    lines.append("  " + DISCLAIMER)
    lines.append("=" * 56 + "\n")

    return "\n".join(lines)


def format_safety_warning(symptoms: List[str]) -> str:
    """Generate a safety warning block for critical symptoms."""
    critical = [s for s in symptoms if s.lower().strip() in CRITICAL_SYMPTOMS]
    warning = SAFETY_WARNING
    if critical:
        warning += f"\n\n🔴 **Critical symptoms detected:** {', '.join(critical)}"
    return warning


def format_general_response(message: str, language: str = "en") -> str:
    """Wrap a general chat message with minimal formatting."""
    return message


def format_fallback() -> str:
    """Return the fallback message when no symptoms could be extracted."""
    return FALLBACK_MESSAGE


def _confidence_bar(confidence: float, width: int = 8) -> str:
    """Generate a visual confidence bar."""
    filled = round(confidence * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def build_json_response(
    intent: str,
    symptoms: List[str],
    session_symptoms: List[str],
    formatted_response: str,
    warnings: Optional[List[str]] = None,
    language: str = "en",
) -> Dict[str, Any]:
    """
    Build the final structured JSON output.

    Returns:
        {
            "intent": str,
            "symptoms": [...],
            "session_symptoms": [...],
            "response": str,
            "warnings": [...],
            "language": str
        }
    """
    return {
        "intent": intent,
        "symptoms": symptoms,
        "session_symptoms": session_symptoms,
        "response": formatted_response,
        "warnings": warnings or [],
        "language": language,
    }
