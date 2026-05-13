import re

from utils.reasoning_log import log_reasoning

def _strip_html_for_context(text: str, max_len: int = 600) -> str:
    if not text:
        return ""
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"\s+", " ", plain).strip()
    if len(plain) > max_len:
        plain = plain[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return plain


def _format_conversation_for_llm(msgs: list, max_turns: int = 8) -> str:
    """Compact transcript for routing / answering (pronouns, follow-ups)."""
    if not msgs:
        return ""
    tail = msgs[-max_turns:] if len(msgs) > max_turns else msgs
    lines = []
    for m in tail:
        role = "User" if m.get("sender") == "user" else "Assistant"
        content = (m.get("message") or "").strip()
        if m.get("sender") != "user":
            content = _strip_html_for_context(content, max_len=700)
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _conversation_cache_suffix(msgs: list) -> str:
    if not msgs:
        return "0"
    blob = "|".join(f"{m.get('sender')}:{(m.get('message') or '')[:160]}" for m in msgs[-8:])
    return str(abs(hash(blob)) % (10**12))


def build_retrieval_query(msg_en: str, conv_block: str, original_msg: str) -> str:
    """Combine recent chat with the latest question so embeddings match follow-ups."""
    base = (msg_en or "").strip() or (original_msg or "").strip()
    if not conv_block.strip():
        return base
    tail = conv_block[-1400:] if len(conv_block) > 1400 else conv_block
    return f"{tail}\n\nCurrent question: {(original_msg or '').strip()}".strip()


user_contexts = {}

def reset_context(ctx):
    ctx["intent"] = None
    ctx["awaiting"] = None
    ctx["data"] = {}
    ctx["last"] = None
    ctx["order_id"] = None


def _normalize_repeated_letters(token: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1", token or "")


def _looks_like_greeting_message(msg: str) -> bool:
    raw = (msg or "").strip().lower()
    if not raw:
        return False
    words = re.findall(r"[a-z]+", raw)
    if not words or len(words) > 3:
        return False
    greeting_set = {"hi", "hii", "hey", "heyy", "hello", "helo", "namaste", "namaskar"}
    for w in words:
        ww = _normalize_repeated_letters(w)
        if ww not in greeting_set:
            return False
    return True


def _is_plausible_order_id(token: str) -> bool:
    """
    Order-id shapes we accept:
    - Numeric Welfog ids (API orderId), typically 7+ digits — use 7..12 only to avoid
      clashing with 6-digit Indian PINs in free text.
    - Alphanumeric refs 6..24 chars with at least one letter and one digit.
    Rejects greetings like 'heyyyyy' (letters only, or too short).
    """
    if not token:
        return False
    t = token.strip()
    if re.fullmatch(r"\d{7,12}", t):
        return True
    if not re.fullmatch(r"[A-Za-z0-9]{6,24}", t):
        return False
    if not re.search(r"[A-Za-z]", t):
        return False
    if not re.search(r"\d", t):
        return False
    return True


def extract_order_id(msg):
    for tk in re.findall(r"\b[A-Za-z0-9]{6,24}\b", msg or ""):
        if _is_plausible_order_id(tk):
            return tk.upper()
    return None

# --- Roman Hindi / Hinglish product understanding (fills gaps when Groq returns wrong intent) ---
_PRODUCT_QUERY_STOPWORDS = frozenset({
    "h", "hai", "he", "ho", "hun", "hoon", "ky", "kya", "ka", "ke", "ki", "ko", "me", "mai", "main",
    "par", "pe", "se", "tak", "kab", "kaise", "kyu", "kyun", "wala", "wale", "wali", "walon",
    "welfog", "app", "website", "online", "pls", "please", "bhai", "yr", "yrr", "yaar", "dear",
    "mil", "milega", "milegi", "milta", "milti", "milen", "sakta", "skta", "sakti", "skti", "sakte",
    "dikha", "dikhaa", "dikhao", "dikhaan", "dikhado", "dikhaana", "dikhaaoo", "de", "do", "dena", "dedo",
    "bata", "batao", "btao", "show", "send", "list", "saare", "sab", "all", "any", "koi", "kuch",
    "chahiye", "chiye", "chaahiye", "lenaa", "lena", "len", "lun", "buy", "need", "want", "looking",
    "for", "the", "a", "an", "is", "are", "there", "have", "has", "stock", "available", "get",
    "categories", "category", "categor", "id", "name",
})

_POLICY_QUESTION_HINTS = frozenset({
    "refund", "return", "payment", "track", "order id", "orderid", "cancel", "policy", "complaint",
    "support", "grievance", "invoice", "delivery time", "shipping time", "damaged", "wrong item",
})


def extract_pincode_from_text(text: str) -> str:
    """First valid 6-digit Indian PIN (leading digit 1–9) in the message, else ''."""
    if not text:
        return ""
    m = re.search(r"\b[1-9]\d{5}\b", text)
    return m.group(0) if m else ""


def _text_is_order_tracking_intent(t: str) -> bool:
    """Track/status of an existing order — not 'can I place an order from X'."""
    tl = f" {t.lower()} "
    markers = (
        "track", "tracking", "order status", "order id", "orderid", "where is my order",
        "where my order", "delivery status", "shipment", "kab aayega", "kab aaega",
        "kab milega", "kahan hai order", "order kahan", "parcel", "package kahan",
        # Additional Hindi/Hinglish patterns for broader matching
        "track ese", "track kar", "ese track", "track kaise", "tracking ese", "track ki",
        "order track", "order ko track", "mera order", "mere order", "order ka status",
        "order status", "mujhe order", "order milega", "order aayega", "iska status",
        "package kahan hai", "parcel kahan hai", "kab deliver", "deliver kab",
    )
    return any(m in tl for m in markers)


def _text_needs_order_id_for_tracking(t: str) -> bool:
    """Return False when the user wants steps / guidance, not an immediate live status lookup."""
    tl = f" {t.lower()} "
    # If they already pasted something that looks like an order id, we should run the API path.
    if extract_order_id(tl):
        return True

    track_core = any(
        x in tl
        for x in (
            "track",
            "tracking",
            "order status",
            "delivery status",
            "shipment",
            "parcel",
            "package",
            "order track",
            "track order",
            "mera order",
            "mere order",
        )
    )
    how_to = any(
        x in tl
        for x in (
            "how to",
            "how can i",
            "how do i",
            "how do we",
            "kaise",
            "kese",
            "kaise kru",
            "kese kru",
            "kaise karu",
            "kese karu",
            "kaise kare",
            "kese kare",
            "kaise krte",
            "kese krte",
            "kaise hot",
            "kese hot",
            " steps",
            "tracking process",
            "track process",
            "order tracking process",
            "tarika",
            "tarike",
            "tareeka",
            "tareeke",
            "guide",
            "batao",
            "btao",
            "bataye",
            "bataiye",
            "tell me",
            "what is the way",
            "kahan se",
            "kaha se",
            "kaha mile",
            "kahan mile",
            "kidhar",
            " tips",
            "tutorial",
        )
    )
    if track_core and how_to:
        return False

    guidance_phrases = (
        "how to track",
        "how can i track",
        "order status please",
        "order status kaise",
        "track order kaise",
        "order track kaise",
        "track karna",
        "tracking kaise",
        "order tracking kaise",
        "tracking process",
        "order tracking process",
        "track karne ka",
        "order track kar",
        "track karne ka tarika",
        "how to track my order",
        "how to check order status",
        "how to check order",
        "kahan se track",
        "kaha se track",
        "kaha se order track",
        "track karne ka",
        "order status dekhen",
        "order id kaha",
        "order id kahan",
        "order id kaise",
        "order id kese",
        "where to find order id",
        "how to find order id",
        "how to get order id",
        "order id nahi pata",
        "order id nahin pata",
        "track ese",
        "track kar",
        "ese track",
        "track kaise",
        "track kese",
        "tracking ese",
        "tracking kese",
        "order track ese",
        "order ko track",
        "track process",
        "track kaise hota",
        "track kese hota",
        "order tracking process",
        "track kaise kru",
        "track kese kru",
        "order track kaise hota hai",
        "order track kese hota hai",
        "how to track",
        "track kaise hota hai",
        "track kese hota hai",
        "order track kaise",
        "order track kese",
    )
    if any(x in tl for x in guidance_phrases):
        return False
    return True


def _text_is_order_id_help_request(t: str) -> bool:
    tl = f" {t.lower()} "
    if not any(x in tl for x in ("order id", "orderid", "order-id")):
        return False
    help_phrases = (
        "kaha", "kahan", "kaise", "kese", "find", "dhoond", "where", "pata", "nahin pata",
        "kahan se", "kaise nikaal", "kaise nikal", "nikal", "nahi pata",
    )
    return any(x in tl for x in help_phrases)


def _text_needs_order_id_for_refund_or_payment(t: str) -> bool:
    """Return False for refund/payment/cancel queries that are about policy/help, not a specific status lookup.
    Works across English, Hindi, Hinglish.
    """
    tl = f" {t.lower()} "
    status_phrases = (
        "status", "track", "tracking", "kab aayega", "kab aaega", "kab milega", "kaha hai", "kaha hai order",
        "kahan hai", "order kahan", "where is my refund", "refund status", "payment status", "transaction status",
        "order status", "current status", "update", "updated",
    )
    help_phrases = (
        "policy", "policies", "kaise", "kya", "process", "procedure", "how to", "kaise kar", "kaise mil", "order cancel",
        "cancel order", "refund kaise", "return kaise", "exchange kaise", "payment kaise", "refund policy", "return policy",
        "exchange policy", "cancel policy", "cancel karna", "refund karna", "return karna", "exchange karna",
        "refund krna", "return krna", "cancel krna", "order cancel kaise", "cancel order kaise", "refund karna hai", "return karna hai",
        "cancel karna hai", "payment kaise", "payment karna",
        # Additional Hindi/Hinglish patterns for policy/help detection
        "kaise kru", "kaise kar sakte", "kaise ho sakta", "process kya hai", "kya process hai",
        "refund process", "cancel process", "payment process", "paise wapsi process", "refund ka procedure",
        "cancel ka tarika", "refund ka tarika", "kaise socha hai", "kaise dekh sakte",
    )
    if any(x in tl for x in help_phrases) and not any(x in tl for x in status_phrases):
        return False
    return True


def _text_has_delivery_or_order_area_intent(t: str) -> bool:
    """
    Delivery serviceability / can I order from this place — Roman Hindi + English.
    Must run BEFORE broad 'product' heuristics so city + delivery does not hit product search.
    """
    tl = f" {t.lower()} "
    if extract_pincode_from_text(t):
        return True
    if any(x in tl for x in [" pincode", " pin code", " pin-code", "zip code", "postal code"]):
        return True
    if any(
        x in tl
        for x in [
            "delivery", "delevery", "delivry", "deliver", "shipping", "courier", "dispatch",
            "ship to", "ship kar",
        ]
    ):
        return True
    if any(
        x in tl
        for x in [
            "order kr", "order kar", "order skt", "order sak", "order skte", "order sakte",
            "order kru", "order karu", "place order", "order dal", "order dunga",
            "mangwa", "mangwa sak", "manga sak", "mangwa skt", "mang skt",
            "aa jayegi", "aa jaayegi", "aayegi", "aayega", "pahuch", "pohuch", "pahuchega",
            "de deg", "de dega", "dedega", "dega na", "milegi delivery", "delivery milegi",
            "service area", "deliver ho", "deliver ho sak", "serviceable",
        ]
    ):
        return True
    if "welfog" in tl and any(x in tl for x in ["use kr", "use kar", "use skt", "use sak", "chal sak", "chalega", "chlega"]):
        if any(x in tl for x in [" se ", " me ", " mein ", "wale", "walo", "city", " pin", " pincode"]):
            return True
    return False


def _text_has_platform_overview_intent(t: str) -> bool:
    """What is on Welfog / how many product types — not a SKU search."""
    tl = t.lower()
    if any(x in tl for x in ["kitne product", "kitne products", "how many product", "how many products"]):
        return True
    if "kya kya" in tl and any(x in tl for x in ["mil", "milega", "milta", "milegi", "milti", "milt", "available"]):
        return True
    if any(x in tl for x in ["what do you sell", "what can i buy", "what all can i", "what is sold"]):
        return True
    if "welfog" in tl and any(x in tl for x in ["kya kya", "kya hai", "kya milta", "about welfog", "pe kya", "par kya", "me kya", "mein kya"]):
        if not _text_has_product_shopping_intent(tl) and "order track" not in tl:
            return True
    return False


def _merge_extracted_pincode(original_msg: str, msg_en: str, ai_data: dict) -> None:
    if not ai_data:
        return
    cur = (ai_data.get("extracted_pincode") or "").strip()
    if cur and re.fullmatch(r"[1-9]\d{5}", cur):
        return
    pin = extract_pincode_from_text(original_msg) or extract_pincode_from_text(msg_en)
    if pin:
        ai_data["extracted_pincode"] = pin


def _looks_like_conversational_followup(original_msg: str, msg_en: str) -> bool:
    """
    Short follow-ups (uska/uske, 'kya karta hai', 'batao') refer to the last topic — not a product SKU search.
    """
    combined = f" {original_msg} {msg_en} ".lower()
    pronoun = (
        " uska ", " uske ", " uski ", " unka ", " unke ", " unki ", " iska ", " iske ", " iski ",
        " inka ", " inke ", " inki ", " yeh ", " ye ", " woh ", " wo ", " is ", " un ",
    )
    explain = (
        "kya karta", "kya karti", "kya karte", " krta ", " krti ", " krte ", " karte ",
        "kaam kya", "kam kya", "what does", "what do ", "how does", "explain", "details",
        "aur bata", "aur btao", "aur batao", " bta ", " btao ", " batao ", "tell me more",
        "iske bare", "iske baare", "uske bare", "uske baare", "iske bar", "uske bar",
    )
    if any(p in combined for p in pronoun) and any(e in combined for e in explain):
        return True
    if any(p in combined for p in pronoun) and any(x in combined for x in (" bta", "btao", "batao", "bata", "detail")):
        return True
    return False


def _text_has_product_shopping_intent(t: str) -> bool:
    """True if message reads like buy/show/availability in English or Roman Hindi."""
    t = f" {t.lower()} "
    non_product_hints = (
        " about ", "department", "departments", "team ", "teams ", "staff ", "company ",
        "policy", "policies", "information", "details",
    )
    if any(h in t for h in non_product_hints):
        return False
    en_markers = (
        "show", "buy", "need", "want", "looking for", " search ", " find ", "price ", " rate ",
        "cheap", "under rs", "under ₹", "purchase", "shop ",
    )
    if any(m in t for m in en_markers):
        return True
    hi_markers = (
        "dikha", "dikhao", "dikh", "dikhaa", "dikhaan",
        "milta", "milti", "milega", "milegi",
        "mil sk", "mil sak", "mile sk", "mile sak",
        "sakta", "skta", "sakti", "skti",
        "chahiye", "chiye", "chaahiye",
        " hai kya", " h kya", " hai ky", " h ky",
        "kidhar", "kahan", "kha se", "kb milega",
    )
    return any(m in t for m in hi_markers)


def _looks_like_browse_all_categories_message(t: str) -> bool:
    t = t.lower()
    return any(
        x in t
        for x in (
            "all categor", "saari categor", "sari categor", "category list", "categories list",
            "list of categor", "konsi categor", "kaun si categor", "browse categor", "har categor",
        )
    )


def _text_has_refund_or_return_intent(t: str) -> bool:
    """Detect refund, return, cancel, exchange intent. Works across English, Hindi, Hinglish."""
    tl = f" {t.lower()} "
    if "refund" in tl:
        return True
    if "return" in tl and any(x in tl for x in ("order", "product", "item", "exchange", "policy", "refund", "cancel", "shipment", "parcel", "purchase")):
        return True
    
    cancel_phrases = (
        "cancel order", "order cancel", "cancel my order", "cancel purchase", "cancel item",
        "cancel kar", "cancel kr", "cancel karna", "cancel ho", "cancel karu",
        # Additional Hindi/Hinglish patterns for cancel/refund/return
        "refund kar", "refund kr", "refund karna", "refund krna", "refund chahiye", "refund chaiye",
        "paise wapsi", "paise vapsi", "paise return", "money back", "return kar", "return kr",
        "return karna", "return chahiye", "order cancel krna", "order cancel karna",
        "order refund", "order return", "exchange karna", "exchange chahiye", "nahin chahiye",
        "nahi chahiye", "order nahin chahiye", "galat mila", "ghalat mila", "cancel order krna",
        "order cancel kaise", "refund kaise", "return kaise", "exchange kaise", "cancel kaise",
        "refund process", "cancel process", "refund kab", "refund kab milega", "order cancel kab",
        "paise kab milenge", "refund kab milega", "order cancel kar do", "order cancel kr do",
        "refund kar do", "refund kr do", "money back le", "order return karna", "order nahin chahiye",
    )
    if any(phrase in tl for phrase in cancel_phrases):
        return True
    return False


def _looks_like_policy_faq_message(t: str) -> bool:
    tl = t.lower()
    return any(h in tl for h in _POLICY_QUESTION_HINTS)


def _looks_like_factual_identity_query(text: str) -> bool:
    """
    Only Welfog-related identity/org questions get strict KB grounding.
    Random person names ("abhishek kon h") must NOT hit the admin-panel fallback.
    """
    t = f" {text.lower()} "
    if "welfog" not in t and "wel fog" not in t:
        return False
    return any(
        h in t
        for h in (
            "who is",
            "who are",
            "who was",
            "kon h",
            "kon hai",
            "kaun h",
            "kaun hai",
            "kisne banaya",
            "founder",
            "co-founder",
            "cofounder",
            "owner",
            "ceo",
            "cto",
            "cfo",
            "funding partner",
            "partner of welfog",
            "about welfog",
            "welfog about",
            "team",
            "staff",
            "department",
        )
    )


def extract_product_search_query(original_msg: str, msg_en: str, ai_search_query=None) -> str:
    """
    Prefer a sensible model-provided search_query; otherwise strip Roman-Hindi filler
    and keep product nouns/brands (e.g. 'cover h ky' -> 'cover').
    """
    sq = (ai_search_query or "").strip()
    if sq and len(sq) >= 2 and sq.lower() not in ("product", "item", "thing", "na", "n/a", "none", "null"):
        return sq
    combined = f"{original_msg} {msg_en}".lower()
    tokens = re.findall(r"[a-z0-9]+", combined)
    kept: list[str] = []
    seen: set[str] = set()
    for w in tokens:
        if len(w) < 2 or w in _PRODUCT_QUERY_STOPWORDS:
            continue
        if w not in seen:
            seen.add(w)
            kept.append(w)
    return " ".join(kept).strip()


def apply_hinglish_product_fixes(original_msg: str, msg_en: str, ai_data: dict) -> None:
    """
    Correct common mis-routes: availability questions in Hinglish must hit product search,
    not categories list or a generic FAQ paragraph.
    """
    if not ai_data:
        return
    intent = ai_data.get("intent")
    if intent == "out_of_domain" or not ai_data.get("is_welfog_related", True):
        return

    if _looks_like_conversational_followup(original_msg, msg_en):
        if ai_data.get("intent") == "product":
            ai_data["intent"] = "general"
            ai_data["search_query"] = ""
    intent = ai_data.get("intent")

    combined = f"{original_msg} {msg_en}"
    comb_low = combined.lower()
    if _text_has_delivery_or_order_area_intent(comb_low):
        if intent == "product":
            ai_data["intent"] = "pincode_check"
            ai_data["search_query"] = ""
        return
    if _text_has_platform_overview_intent(comb_low):
        if intent == "product":
            ai_data["intent"] = "general"
            ai_data["search_query"] = ""
        return
    extracted = extract_product_search_query(original_msg, msg_en, ai_data.get("search_query"))

    if intent == "categories" and extracted and not _looks_like_browse_all_categories_message(combined):
        tl = combined.lower()
        short_concrete = len(re.findall(r"[a-z0-9]+", tl)) <= 5 and "categor" not in tl
        if _text_has_product_shopping_intent(combined) or short_concrete:
            ai_data["intent"] = "product"
            ai_data["search_query"] = extracted
            return

    if (
        intent == "general"
        and extracted
        and _text_has_product_shopping_intent(combined)
        and not _looks_like_conversational_followup(original_msg, msg_en)
    ):
        if not _looks_like_policy_faq_message(combined):
            ai_data["intent"] = "product"
            ai_data["search_query"] = extracted
            return

    if intent == "product":
        cur = (ai_data.get("search_query") or "").strip()
        if not cur or len(cur) < 2 or cur.lower() in ("product", "item", "thing"):
            if extracted:
                ai_data["search_query"] = extracted


