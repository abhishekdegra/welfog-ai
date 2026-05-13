import json
import os
import re
import time

import requests

from services.kb_service import get_runtime_knowledge_files
from utils.reasoning_log import log_reasoning

def ai_brain_route(user_msg, conversation_context: str = ""):
    """
    Step 1: Use Groq to understand the user message (any language),
    decide intent + which knowledge files should be used for grounding.
    """
    try:
        groq_api_key = os.getenv("GROQ_API_KEY") 
        if not groq_api_key:
            print("⚠️ ERROR: Groq API Key nahi mili! .env file check kar.")
            return None

        url = "https://api.groq.com/openai/v1/chat/completions"
        model_name = "llama-3.1-8b-instant" 

        kb_keys_list = ", ".join([f'"{k}"' for k in get_runtime_knowledge_files().keys()])
        system_prompt = f"""You are 'Welfog AI', an intelligent e-commerce assistant for Welfog.
Your job in this step is ONLY routing + understanding. Return ONLY a valid JSON object.

Available knowledge file keys (choose from these only):
[{kb_keys_list}]

JSON SCHEMA:
{{
  "reasoning": "Analyze the user's intent step-by-step.",
  "intent": "product" | "order" | "refund" | "payment" | "seller" | "pincode_check" | "deals" | "categories" | "category_feed" | "general" | "out_of_domain",
  "kb_keys": ["One or more knowledge keys to use for grounding"],
  "search_query": "Clean product name if product intent, else empty",
  "extracted_pincode": "Extract the 6-digit PIN code if the user provides one, else empty string",
  "needs_order_id": true/false,
  "is_welfog_related": true/false
}}

CRITICAL ROUTING RULES:
1) DEALS/OFFERS: If user asks for deals/offers/discount/today deals -> intent="deals", kb_keys MUST include "welfog_api".
2) CATEGORIES LIST: ONLY if user clearly wants the full department list ("all categories", "category list", "saari category", "browse all categories") -> intent="categories", kb_keys MUST include "welfog_api".
   NEVER use intent="categories" for a specific item/brand/stock question (e.g. Roman Hindi "durex h ky", "cover hai kya", "pen milta hai") — those are PRODUCT searches.
2b) CATEGORY-WISE FEED: If user asks "category wise products", "home page products", "browse by category", "trending by category" -> intent="category_feed", kb_keys MUST include "welfog_api".
3) PINCODE / DELIVERY AREA: If the user asks whether Welfog delivers to a place, if they can order from a city/area, serviceability, or uses Roman Hindi such as "delivery chahiye", "X se order kr sakta", "aa jayegi delivery", "de dega na delivery", "pincode", or gives a 6-digit Indian PIN -> intent="pincode_check". Put the PIN in extracted_pincode when present, else "". kb_keys MUST include "shipping", "faqs", and usually "company".
   Do NOT use intent="product" when there is no specific product to buy—only location/delivery/serviceability.
4) PLATFORM OVERVIEW: Broad questions like "welfog me kya kya milta", "kitne products", what Welfog sells in general -> intent="general", kb_keys include "company", "faqs". Not a product SKU search.
4b) COMPANY / TEAM / POLICY FROM KB: Questions about departments, staff, roles, "kya karta hai", office info, policies, FAQs -> intent="general". Choose kb_keys that match topic filenames when obvious; otherwise include several customer-facing keys (not only welfog_api). NEVER answer these as intent="product" unless they clearly ask to buy a physical SKU.
5) PRODUCT: Any specific shopping item question in ANY language or Roman Hindi/Hinglish: "show/buy/need", "dikhao/dikha", "milega/milta", "sakta/skta", "chahiye", "hai kya/h ky", "available", brand or product name alone -> intent="product".
   Set search_query to the core product/brand keyword in English (e.g. "cover h ky" -> "cover", "durex dikha" -> "durex", "pen mil sakta" -> "pen"). Strip question/filler words only.
   kb_keys include "welfog_api" and optionally "faqs".
6) FOLLOW-UPS: If the latest message uses pronouns (uska/uske/iska) or "yeh kya karta" without naming a product, use RECENT CONVERSATION to infer the subject. If it continues a non-shopping explanation (e.g. a department), intent="general" and search_query="".
7) ORDER/REFUND/PAYMENT: Live tracking ("where is my order", status, kab aayega) -> intent="order" (or refund/payment) with needs_order_id=true unless the message clearly asks ONLY for steps/help ("how to track", "order id kahan", "kaise/kese track", Roman Hindi) — then needs_order_id=false and kb_keys include shipping/faqs/welfog_api.
8) OUT OF DOMAIN: unrelated chit-chat, homework, sports, politics, other companies -> is_welfog_related=false, intent="out_of_domain".
Return JSON only."""
        
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }

        user_payload = user_msg
        if (conversation_context or "").strip():
            user_payload = (
                "RECENT CONVERSATION (use to resolve pronouns and follow-ups):\n"
                f"{conversation_context.strip()}\n\n"
                f"LATEST USER MESSAGE:\n{user_msg}"
            )

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ],
            "response_format": {"type": "json_object"}, 
            "temperature": 0.0,
            "max_tokens": 220
        }
        out = _groq_json_with_retry(url, headers, payload, timeout_sec=10, max_attempts=3)
        if out:
            log_reasoning(out.get("reasoning") or "Routing completed.")
        return out
            
    except Exception as e:
        print("AI Brain Error:", e)
        return None


def _extract_retry_wait_seconds(error_text: str) -> float:
    if not error_text:
        return 1.5
    m = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", error_text, flags=re.IGNORECASE)
    if not m:
        return 1.5
    try:
        return max(0.5, min(8.0, float(m.group(1)) + 0.2))
    except Exception:
        return 1.5


def _groq_json_with_retry(url, headers, payload, timeout_sec=12, max_attempts=3):
    """
    Retry on transient Groq failures (rate-limit / invalid-json generation).
    Returns parsed JSON dict or None.
    """
    req = dict(payload or {})
    for attempt in range(1, max_attempts + 1):
        try:
            res = requests.post(url, headers=headers, json=req, timeout=timeout_sec)
            if res.status_code == 200:
                body = res.json()
                content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
                if not content:
                    continue
                try:
                    return json.loads(content)
                except Exception:
                    # one more try with smaller output budget
                    req["max_tokens"] = max(120, int(req.get("max_tokens", 300) * 0.7))
                    time.sleep(0.5)
                    continue

            text = res.text or ""
            low = text.lower()
            is_rate = (res.status_code == 429) or ("rate_limit_exceeded" in low)
            is_json_fail = ("json_validate_failed" in low) or ("failed to generate json" in low)
            if is_rate and attempt < max_attempts:
                wait_sec = _extract_retry_wait_seconds(text)
                log_reasoning(f"Groq rate-limited; retrying in {wait_sec:.1f}s (attempt {attempt}/{max_attempts})")
                time.sleep(wait_sec)
                continue
            if is_json_fail and attempt < max_attempts:
                req["max_tokens"] = max(120, int(req.get("max_tokens", 300) * 0.7))
                time.sleep(0.5)
                continue
            print(f"Groq API Error: {text}")
            return None
        except Exception as e:
            if attempt < max_attempts:
                time.sleep(0.7 * attempt)
                continue
            print("AI Brain Error:", e)
            return None
    return None


def ai_brain_answer(user_msg, kb_context, conversation_context: str = ""):
    """
    Step 2: Use Groq with selected KB context to generate final answer JSON.
    """
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("⚠️ ERROR: Groq API Key nahi mili! .env file check kar.")
            return None

        url = "https://api.groq.com/openai/v1/chat/completions"
        model_name = "llama-3.1-8b-instant"

        system_prompt = f"""You are 'Welfog AI', an intelligent e-commerce assistant for Welfog.
Analyze the user message semantically, use the KNOWLEDGE BASE as the source of truth, and return ONLY a valid JSON object.

WELFOG KNOWLEDGE BASE (Selected + relevant excerpts):
\"\"\"
{kb_context}
\"\"\"

JSON SCHEMA:
{{
  "reasoning": "Short reasoning (1-3 lines).",
  "intent": "product" | "order" | "refund" | "payment" | "seller" | "pincode_check" | "deals" | "categories" | "category_feed" | "general" | "out_of_domain",
  "search_query": "Clean product name if product intent, else empty",
  "extracted_pincode": "Extract the 6-digit PIN code if the user provides one, else empty string",
  "needs_order_id": true/false,
  "is_welfog_related": true/false,
  "response": "Final answer: complete enough to satisfy the question. If intent=categories or deals, guide user what you are showing."
}}

RULES:
- Answer ONLY what the user asked; keep the "response" concise (no extra sections, no unrelated marketing).
- Follow the API PLAYBOOK instructions if present in KB (source=welfog_api).
- Do NOT invent products, prices, categories, or deals. If intent is "product", give a SHORT line that results are being shown — do NOT tell the user to only visit the website/app instead of searching.
- KNOWLEDGE ANSWERS: Read the excerpts fully. If the user asks for a list (e.g. department names, team list), output the actual names/details from the text — do NOT reply with only a document title, tag line, or \"According to...\" meta phrase. Use bullets or short sentences when listing multiple items.
- If the requested detail is not present in KB context, clearly say it is not available right now instead of guessing.
- If the user asks about refund, return, cancel, payment, or order policy, answer directly and keep it concise. Do not add unrelated paragraphs or marketing text.
- ORDER TRACKING: If they ask HOW/WHERE to track (steps, process, "kaise/kese", order id location) and did not give an order id, set needs_order_id=false and give short numbered steps (app/website → My Orders → SMS/email for Order ID). If they want live status for their order and have or will share an id, needs_order_id=true.
- Roman Hindi / Hinglish: specific item availability ("milega", "hai kya") -> intent "product" + search_query. Delivery-to-a-place / can-I-order-from-here / PIN-only messages -> intent "pincode_check" (ask for 6-digit PIN if missing). General "what is on Welfog" -> intent "general", not product.
- FOLLOW-UPS: If RECENT CONVERSATION is provided, resolve pronouns (uska/uske/iska/yeh) from the last turns; keep intent "general" when continuing an explanatory topic (e.g. what a department does) unless the user clearly switches to buying a product.
- Keep response clean and helpful; Hinglish/Hindi allowed based on user language."""

        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }

        user_payload = user_msg
        if (conversation_context or "").strip():
            user_payload = (
                "RECENT CONVERSATION:\n"
                f"{conversation_context.strip()}\n\n"
                f"LATEST USER MESSAGE:\n{user_msg}"
            )

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 380
        }
        out = _groq_json_with_retry(url, headers, payload, timeout_sec=12, max_attempts=3)
        if out:
            log_reasoning(out.get("reasoning") or "Answer generation completed.")
        return out
    except Exception as e:
        print("AI Brain Error:", e)
        return None
