from langdetect import detect
from deep_translator import GoogleTranslator

from utils.reasoning_log import log_reasoning

ALLOWED_LANGS = ["en", "hi", "gu", "pa", "ta", "te", "mr", "ml", "kn", "bn", "ur", "hinglish"]


def detect_language(msg):
    """
    Detect language for reply translation.

    Goal: The reply should match the language the customer used.
    Fix: English messages like "how can i track my order" should stay `en`.
    """
    if not msg or len(msg.strip()) == 0:
        return "en"

    try:
        # 1) Strong signal: Devanagari script => Hindi
        if any("\u0900" <= ch <= "\u097F" for ch in msg):
            return "hi"

        lang = detect(msg)
        msg_lower = msg.lower()

        # 2) Roman-Hinglish markers (verbs/particles).
        # Intentionally do NOT include plain English nouns like order/track/refund/cancel
        # to avoid false Hindi on English-only queries.
        hinglish_markers = [
            "kya",
            "hai",
            "kar",
            "kr",
            "nahi",
            "kyu",
            "kaise",
            "kese",
            "krna",
            "kru",
            "iske",
            "iska",
            "mujhe",
            "mere",
            "apna",
            "pata",
            "kahan",
            "kaha",
            "batao",
            "btao",
            "samjho",
            "samajh",
            "dekh",
            "suna",
            "sunai",
            "chahiye",
            # also common conjugations
            "krta",
            "karti",
            "karte",
            "kru",
            "skta",
            "sakta",
        ]

        def _has_marker(w: str) -> bool:
            # lightweight boundary check; avoids matching inside English words
            return f" {w} " in f" {msg_lower} " or msg_lower.startswith(w) or msg_lower.endswith(w)

        hinglish_count = sum(1 for w in hinglish_markers if _has_marker(w))

        # Trust langdetect when it confidently says Hindi.
        if lang == "hi":
            return "hi"

        # Force Hindi if Hinglish signals are present.
        # >=2 markers => definitely Hinglish.
        if hinglish_count >= 2:
            return "hi"

        # Single strong marker => treat as Hinglish (langdetect may output random code).
        strong_single_markers = [
            "kaise",
            "kese",
            "kr",
            "kar",
            "nahi",
            "chahiye",
            "mujhe",
            "mere",
            "apna",
            "batao",
            "btao",
            "samjho",
            "samajh",
            "dekh",
            "suna",
            "sunai",
        ]
        if any(_has_marker(w) for w in strong_single_markers):
            return "hi"

        return lang if lang in ALLOWED_LANGS else "en"
    except Exception as e:
        log_reasoning(f"Language detection error: {e}. Fallback heuristic...")
        m = msg.lower()
        if any(x in m for x in ["kaise", "kese", "kr ", "kar ", "nahi", "mujhe", "mere", "apna", "batao", "btao"]):
            return "hi"
        return "en"


def to_en(text):
    """Translate text to English. Falls back to original if translation fails."""
    if not text or len(text.strip()) == 0:
        return text
    try:
        result = GoogleTranslator(source='auto', target='en').translate(text)
        if result and isinstance(result, str) and len(result.strip()) > 0:
            return result
    except Exception as e:
        log_reasoning(f"Translation error (to_en): {e}. Using original text for processing.")
    return text


def to_user(text, lang):
    if lang == "en":
        return text
    try:
        return GoogleTranslator(source='en', target=lang).translate(text)
    except Exception:
        return text
