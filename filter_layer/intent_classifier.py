"""
filter_layer/intent_classifier.py
Classifies user input as GENERAL chat or CLINICAL (medical) query.
"""

import re
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keywords that indicate pure general chat (non-clinical)
# ---------------------------------------------------------------------------
GENERAL_CHAT_KEYWORDS = {
    "hi", "hello", "hey", "helo", "hii", "namaste", "namaskar", "kem cho",
    "kya haal", "whats up", "what's up", "sup",
    "who are you", "what are you", "what is this", "what do you do",
    "how does this work", "help", "about", "info", "information",
    "what can you do", "tell me about",
    "bye", "goodbye", "good bye", "see you", "cya", "ok thanks",
    "thank you", "thanks", "thx", "okay", "ok",
    "good morning", "good evening", "good night",
    "how are you", "i am fine", "i'm fine", "all good",
    "nice", "cool", "great", "awesome",
    "what is astra", "who built this", "are you a doctor",
}

# Clinical trigger words — any of these strongly indicates a medical query
CLINICAL_TRIGGER_WORDS = {
    "pain", "ache", "fever", "cough", "cold", "nausea", "vomiting", "diarrhea",
    "headache", "throat", "breathing", "chest", "dizzy", "dizziness", "fatigue",
    "weakness", "rash", "itch", "swelling", "burn", "burning", "infection",
    "symptom", "symptoms", "disease", "sick", "ill", "illness", "diagnose",
    "treatment", "medicine", "medication", "relief", "hurt", "hurting",
    "suffering", "problem", "issue", "condition", "health", "body",
    "stomach", "abdomen", "back", "joint", "muscle", "skin", "eye", "ear",
    "feel", "feeling", "having", "suffering from", "experiencing",
    # Hindi/Gujarati hints that survived normalization
    "bukhar", "dard", "khansi", "tav", "ulti", "chakkar",
}

# Friendly responses for general chat
GENERAL_RESPONSES = {
    "greeting": (
        "👋 Hello! I'm Astra, your AI-powered clinical assistant. "
        "Describe your symptoms in any language (English, Hindi, Gujarati) "
        "and I'll help analyze them for you."
    ),
    "farewell": (
        "👋 Take care! If you experience any symptoms, don't hesitate to come back. "
        "Stay healthy! 🌿"
    ),
    "identity": (
        "🩺 I'm **Astra A0**, a multilingual clinical intelligence engine. "
        "I can understand symptoms in English, Hindi, Gujarati, and Hinglish, "
        "and provide Ayurvedic insights. Tell me how you're feeling!"
    ),
    "default": (
        "😊 I'm here to help with your health queries. "
        "Please describe your symptoms (e.g., 'I have fever and headache') "
        "and I'll provide a clinical analysis."
    ),
}


def _is_farewell(text: str) -> bool:
    farewell_words = {"bye", "goodbye", "good bye", "see you", "cya", "ok thanks", "thank you", "thanks"}
    tokens = set(text.lower().split())
    return bool(tokens & farewell_words)


def _is_identity_query(text: str) -> bool:
    triggers = {"who are you", "what are you", "what is this", "what is astra",
                "what do you do", "are you a doctor", "who built this", "about"}
    text_lower = text.lower()
    return any(t in text_lower for t in triggers)


def classify_intent(text: str, extracted_symptoms: Optional[List[str]] = None) -> Tuple[str, str]:
    """
    Classify intent of user input.

    Args:
        text: Raw or normalized user input
        extracted_symptoms: Symptoms already extracted (can inform classification)

    Returns:
        Tuple of (intent, response_message)
        - intent: "general" | "clinical"
        - response_message: Pre-built reply for general chat, "" for clinical
    """
    if not text:
        return "general", GENERAL_RESPONSES["default"]

    text_lower = text.lower().strip()

    # If symptoms were already extracted, it's definitely clinical
    if extracted_symptoms:
        logger.debug("Clinical intent confirmed by extracted symptoms: %s", extracted_symptoms)
        return "clinical", ""

    # Check if the entire input is a known general phrase
    if text_lower in GENERAL_CHAT_KEYWORDS:
        if _is_farewell(text_lower):
            return "general", GENERAL_RESPONSES["farewell"]
        if text_lower in {"hi", "hello", "hey", "helo", "hii", "namaste", "namaskar", "kem cho"}:
            return "general", GENERAL_RESPONSES["greeting"]
        return "general", GENERAL_RESPONSES["default"]

    # Check for identity queries
    if _is_identity_query(text_lower):
        return "general", GENERAL_RESPONSES["identity"]

    # Check for farewell
    if _is_farewell(text_lower):
        return "general", GENERAL_RESPONSES["farewell"]

    # Check for clinical trigger words in the text
    tokens = set(re.findall(r'\b\w+\b', text_lower))
    if tokens & CLINICAL_TRIGGER_WORDS:
        logger.debug("Clinical trigger words found: %s", tokens & CLINICAL_TRIGGER_WORDS)
        return "clinical", ""

    # If input is very short and has no clinical markers, treat as general
    if len(text_lower.split()) <= 2:
        return "general", GENERAL_RESPONSES["default"]

    # Default: treat longer unknown inputs as clinical (better safe than sorry)
    return "clinical", ""
