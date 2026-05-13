"""Public chat UI and JSON APIs."""
import json
import re
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

from services.ai_service import ai_brain_answer, ai_brain_route
from services.embedding import GREETINGS, GREETINGS_VECS, model
from services.kb_service import (
    INTERNAL_KB_KEYS,
    best_kb_hit,
    keyword_kb_hit,
    direct_kb_search,
    ensure_knowledge_cache_fresh,
    get_customer_kb_keys,
    get_knowledge_context,
    smart_instant_router,
    sysmsg,
    _KB_SNAPSHOT,
)
from services.mysql_service import (
    db_get_recent_messages,
    db_store_message,
    generate_chat_token,
    get_mysql_connection,
)
from services.translation_service import detect_language, to_en, to_user
from services.welfog_api import (
    _normalize_color,
    check_pincode_delivery,
    fetch_api,
    fetch_category_wise_feed,
    fetch_nav_categories,
    fetch_order_tracking,
    fetch_products_from_api,
    format_order_tracking_reply,
    fetch_today_deals,
    get_category_id_from_text,
)
from utils.helpers import (
    _conversation_cache_suffix,
    _format_conversation_for_llm,
    _looks_like_browse_all_categories_message,
    _looks_like_conversational_followup,
    _looks_like_factual_identity_query,
    _looks_like_greeting_message,
    _merge_extracted_pincode,
    _text_has_delivery_or_order_area_intent,
    _text_has_platform_overview_intent,
    _text_has_product_shopping_intent,
    _text_has_refund_or_return_intent,
    _text_is_order_id_help_request,
    _text_is_order_tracking_intent,
    _text_needs_order_id_for_refund_or_payment,
    _text_needs_order_id_for_tracking,
    apply_hinglish_product_fixes,
    build_retrieval_query,
    extract_order_id,
    reset_context,
    user_contexts,
)
from utils.reasoning_log import log_reasoning
from utils.cache import _cache_get, _cache_set

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/")
def home():
    return render_template("index.html")


@chat_bp.route("/api/chat/new", methods=["POST"])
def new_chat_reset():
    user_id = request.remote_addr
    if user_id in user_contexts:
        reset_context(user_contexts[user_id])
    return jsonify({"status": "cleared"})

@chat_bp.route("/api/chats", methods=["GET"])
def get_history():
    user_id = request.remote_addr
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    conn = get_mysql_connection()
    if not conn:
        return jsonify([])
    try:
        with conn.cursor() as cur:
            # List only sessions that still have rows in `chats` with real JSON messages.
            # Otherwise clearing `chats` in phpMyAdmin leaves "ghost" titles from `chat_sessions`.
            cur.execute(
                """
                SELECT DISTINCT
                    COALESCE(cs.chat_token, CAST(cs.id AS CHAR)) AS chat_id,
                    cs.title,
                    cs.created_at
                FROM chat_sessions cs
                INNER JOIN chats c ON (
                    c.chat_token = cs.chat_token
                    OR c.chat_id = cs.chat_token
                    OR CAST(c.chat_id AS CHAR) = CAST(cs.id AS CHAR)
                )
                WHERE cs.user_id = %s
                  AND cs.created_at >= %s
                  AND c.chat_data IS NOT NULL
                  AND CHAR_LENGTH(TRIM(c.chat_data)) > 5
                ORDER BY cs.created_at DESC
                """,
                (user_id, seven_days_ago),
            )
            rows = cur.fetchall()
        chats = []
        for r in rows:
            ts = r["created_at"]
            date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts).split(" ")[0]
            chats.append({"chat_id": r["chat_id"], "title": r["title"], "date_str": date_str})
        return jsonify(chats)
    except Exception as e:
        print(f"❌ get_history MySQL error: {e}")
        return jsonify([])
    finally:
        conn.close()

@chat_bp.route("/api/chat/delete/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    user_id = request.remote_addr
    conn = get_mysql_connection()
    if not conn:
        return jsonify({"error": "database_unreachable", "message": "Database unavailable."}), 503
    try:
        sid = str(chat_id)
        numeric_chat_id = int(sid) if sid.isdigit() and len(sid) < 20 else None
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, chat_token FROM chat_sessions "
                "WHERE (chat_token = %s OR id = %s) AND user_id = %s LIMIT 1",
                (chat_id, numeric_chat_id if numeric_chat_id is not None else -1, user_id),
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "not_found", "message": "Chat not found or not owned by you."}), 404

            session_id = row["id"]
            token = row["chat_token"]
            cur.execute(
                "DELETE FROM chats WHERE chat_token = %s OR chat_id = %s OR chat_id = %s OR chat_id = %s",
                (token, token, str(session_id), session_id),
            )
            cur.execute("DELETE FROM chat_sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
        conn.commit()
        return jsonify({"status": "deleted"})
    except Exception as e:
        conn.rollback()
        print(f"❌ delete_chat MySQL error: {e}")
        return jsonify({"error": "database_error", "message": "Unable to delete chat."}), 500
    finally:
        conn.close()

@chat_bp.route("/api/chat/messages/<chat_id>", methods=["GET"])
def get_messages(chat_id):
    conn = get_mysql_connection()
    if not conn:
        return jsonify([])  # Agar DB connect na ho toh khali list bhej do
        
    try:
        cursor = conn.cursor()
        numeric_chat_id = int(chat_id) if chat_id.isdigit() and len(chat_id) < 12 else None
        if numeric_chat_id is not None:
            cursor.execute(
                "SELECT chat_data FROM chats WHERE chat_token = %s OR chat_id = %s LIMIT 1",
                (chat_id, numeric_chat_id),
            )
        else:
            cursor.execute("SELECT chat_data FROM chats WHERE chat_token = %s LIMIT 1", (chat_id,))
        row = cursor.fetchone()

        if not row or not row.get("chat_data"):
            return jsonify({"error": "not_found"}), 404

        try:
            chat_data = json.loads(row["chat_data"])
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            return jsonify({"error": "invalid_data"}), 500

        if isinstance(chat_data, dict):
            chat_data = [chat_data]
        elif not isinstance(chat_data, list):
            return jsonify({"error": "not_found"}), 404

        if len(chat_data) == 0:
            return jsonify({"error": "not_found"}), 404

        msgs = []
        for item in chat_data:
            if not isinstance(item, dict):
                continue
            msgs.append({
                "sender": item.get("sender"),
                "message": item.get("text") if item.get("text") is not None else item.get("message"),
            })

        return jsonify(msgs)
        
    except Exception as e:
        print(f"❌ MySQL Fetch Error: {e}")
        return jsonify([])
    finally:
        if conn:
            conn.close()


@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message", "").strip()
    user_id = request.remote_addr
    # Auto-sync KB on every request (handles add/update/delete .txt files dynamically)
    ensure_knowledge_cache_fresh()
    
    # 🔥 FIX 1: Frontend se chat_id fetch karo
    current_chat_id = data.get("chat_id")

    if user_id not in user_contexts:
        user_contexts[user_id] = {"intent": None, "awaiting": None, "data": {}, "last": None, "order_id": None}

    ctx = user_contexts[user_id]
    
    # Naya chat session MySQL chat_sessions me — yahi id `chats.chat_id` se judi rahegi
    if not current_chat_id:
        title = (user_msg[:30] + "...") if len(user_msg) > 30 else (user_msg or "New chat")
        current_chat_id = generate_chat_token()
        conn = get_mysql_connection()
        if not conn:
            # Graceful fallback: allow chat to work even if MySQL is down.
            # History/sidebar persistence won't work until DB is running.
            log_reasoning("MySQL unavailable; starting ephemeral chat session (no persistence).")
            conn = None
        try:
            if conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO chat_sessions (user_id, title, chat_token) VALUES (%s, %s, %s)",
                        (user_id, title, current_chat_id),
                    )
                conn.commit()
        except Exception as e:
            print(f"❌ New chat session insert error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
        
    # 🔥 FIX 3: User ka message database me save karo
    db_store_message(current_chat_id, "user", user_msg)

    original_msg = user_msg.strip()
    lang = detect_language(original_msg)
    
    msg_en = to_en(original_msg).lower().strip() if lang != "en" else original_msg.lower().strip()

    recent_msgs = db_get_recent_messages(current_chat_id, 12)
    conv_for_llm = _format_conversation_for_llm(recent_msgs)
    retrieval_query = build_retrieval_query(msg_en, conv_for_llm, original_msg)
    conv_sig = _conversation_cache_suffix(recent_msgs)

    intent = "general"
    search_query = ""
    is_welfog = True
    ai_response_text = ""
    ai_data = {}
    extracted_id = None

    # 🔥 FIX 4: Smart helper jo har reply ko (zarurat par) translate, save aur return karega.
    def send_reply(text_data, lang_code):
        """
        - Plain text replies -> user language me translate.
        - Structured HTML replies (product/deal cards etc.) -> as‑is, so CSS/markup multilingual me bhi break na ho.
        """
        if lang_code != "en" and isinstance(text_data, str) and any(tag in text_data for tag in ("<div", "<img", "<button", "<span", "<style")):
            # HTML content ko translate nahi karna, warna tags toot jate hain (non-English me layout bigadta hai)
            final_output = text_data
        else:
            final_output = to_user(text_data, lang_code)
        db_store_message(current_chat_id, "bot", final_output)
        return jsonify({"chat_id": current_chat_id, "type": "text", "data": final_output})


    # 1) STRICT STATE LOCKS (awaiting loops)
    if ctx.get("awaiting") == "order_id":
        if original_msg.lower() in ["cancel", "stop", "exit", "no"]:
            log_reasoning("User cancelled order-id collection state.")
            reset_context(ctx)
            return send_reply(sysmsg("cancelled"), lang)

        if _text_is_order_id_help_request(original_msg) or _text_is_order_id_help_request(msg_en):
            guidance = sysmsg("order_id_help") or (
                "Your Order ID appears in the confirmation email, SMS, or on your Welfog account orders page. "
                "Please share it when you're ready, or type 'cancel' to ask something else."
            )
            return send_reply(guidance, lang)

        extracted_id = extract_order_id(msg_en)
        if extracted_id:
            log_reasoning("Awaiting order-id state resolved with a valid ID.")
            intent = ctx.get("last") or "order"
            ctx["order_id"] = extracted_id
            ctx["awaiting"] = None
            ai_data = {"intent": intent, "is_welfog_related": True, "needs_order_id": True} 
        else:
            log_reasoning("Awaiting order-id state: user message not a valid order ID.")
            return send_reply(sysmsg("ask_order_id_generic"), lang)
    
    elif ctx.get("awaiting") == "category_select":
        if original_msg.lower() in ["cancel", "stop", "exit", "no"]:
            reset_context(ctx)
            return send_reply(sysmsg("cancelled"), lang)

        # If user asks a fresh/general question while awaiting category id,
        # unlock this state and continue with normal routing.
        if _looks_like_browse_all_categories_message(msg_en) or "department" in msg_en or "staff" in msg_en:
            ctx["awaiting"] = None

        # Try parse category id (e.g., "16", "id 16", "category 16")
        cat_id = get_category_id_from_text(msg_en, ctx=ctx)
        if not cat_id:
            m = re.search(r"\b(\d{1,5})\b", msg_en)
            if m:
                cat_id = m.group(1)

        color = _normalize_color(msg_en)
        if ctx.get("awaiting") == "category_select" and not cat_id:
            return send_reply(sysmsg("ask_category_select"), lang)

        intent = "product"
        search_query = ""  # category browse
        ai_data = {"intent": "product", "is_welfog_related": True, "search_query": ""}
        ctx["awaiting"] = None
        ctx["data"]["selected_category_id"] = cat_id
        ctx["data"]["selected_color"] = color

    # 2. NORMAL FLOW
    else:
        # If user directly says a category name ("electronics ke products dikhao"),
        # auto resolve category id and show products without asking for id.
        auto_cat_id = get_category_id_from_text(msg_en, ctx=ctx)
        if auto_cat_id and any(w in msg_en for w in ["category", "products", "product", "items", "dikhao", "show", "list"]):
            ctx.setdefault("data", {})
            ctx["data"]["selected_category_id"] = auto_cat_id
            ctx["data"]["selected_color"] = _normalize_color(msg_en)
            intent = "product"
            search_query = ""  # category browse
            ai_data = {"intent": "product", "is_welfog_related": True, "search_query": ""}

        # FAST GREETING CHECK (precomputed)
        if _looks_like_greeting_message(original_msg) or any(original_msg.lower() == g for g in GREETINGS) or cosine_similarity(model.encode([msg_en]), GREETINGS_VECS).max() > 0.75:
            log_reasoning("Greeting detected; responding with greeting template.")
            reset_context(ctx)
            return send_reply(sysmsg("greeting"), lang)

        # ================= ⚡ 1. SMART INSTANT ROUTER (0.1s Execution) ⚡ =================
        fast_result = smart_instant_router(original_msg, msg_en)
        
        if fast_result:
            if fast_result["action"] == "reject" or fast_result["action"] == "text":
                log_reasoning(f"Fast router action='{fast_result['action']}'")
                reset_context(ctx)
                return send_reply(fast_result["data"], lang)
                
            elif fast_result["action"] == "product":
                reset_context(ctx)
                intent = "product"
                search_query = fast_result["query"]
                is_welfog = True
                
            elif fast_result["action"] == "ask_order_id":
                log_reasoning("Fast router requires order id for order/refund/payment flow.")
                ctx["intent"] = fast_result["intent"]
                ctx["last"] = fast_result["intent"]
                ctx["awaiting"] = "order_id"
                return send_reply(sysmsg("ask_order_id_for_intent", intent=fast_result["intent"]), lang)
                
            elif fast_result["action"] == "direct_order_id":
                log_reasoning("Single-token message matched strict order-id pattern.")
                extracted_id = fast_result["order_id"]
                intent = ctx.get("last") or "order"
                ctx["order_id"] = extracted_id
                ctx["awaiting"] = None
                ai_data = {"intent": intent, "is_welfog_related": True, "needs_order_id": True}
                
        else:
            # ================= 🐢 2. AI BRAIN (Only for complex questions) 🐢 =================
            # kb_match = direct_kb_search(msg_en)
            # if kb_match:
            #     reset_context(ctx)
            #     return send_reply(kb_match, lang)
            
            if ctx.get("awaiting") == "order_id" and not extracted_id:
                reset_context(ctx)

            # ================= 📚 KB-FIRST PASS (no hardcoding) =================
            # If answer exists in ANY admin-added knowledge file, prefer that first.
            # Skip this for shopping/product/deals/category flows where API results are expected.
            comb_low = f"{original_msg} {msg_en}".lower()
            kb_first_allowed = not (
                _text_has_product_shopping_intent(comb_low)
                or _text_has_delivery_or_order_area_intent(comb_low)
                or any(w in f" {msg_en} " for w in ["deal", "deals", "offer", "offers", "discount", "today deal"])
                or _looks_like_browse_all_categories_message(msg_en)
                or _text_is_order_tracking_intent(comb_low)
            )
            if kb_first_allowed:
                hit = best_kb_hit(retrieval_query, keys=get_customer_kb_keys(), min_score=0.22)
                if not hit:
                    hit = keyword_kb_hit(retrieval_query, keys=get_customer_kb_keys(), min_hits=2)
                if hit:
                    score_str = f"{hit['score']:.2f}" if isinstance(hit.get("score"), (int, float)) else str(hit.get("score"))
                    log_reasoning(f"KB-first matched source={hit['source']} score={score_str}")
                    # Use Groq once to turn the chunk into a proper answer (and handle follow-ups).
                    kb_context = f"[source={hit['source']} score={score_str}] {hit['chunk']}"
                    ai_data = ai_brain_answer(original_msg, kb_context, conv_for_llm) or {}
                    ai_data.setdefault("intent", "general")
                    ai_data.setdefault("is_welfog_related", True)
                    intent = ai_data.get("intent", "general")
                    is_welfog = ai_data.get("is_welfog_related", True)
                    search_query = ai_data.get("search_query") or msg_en
                    ai_response_text = ai_data.get("response", "")
                    # Short-circuit to execution section with a grounded answer.
                    computed_fresh = False
                    cached_ai = None

            # PERF: short-term response cache (skip Groq entirely on repeats)
            resp_cache_key = f"resp::{_KB_SNAPSHOT}::{current_chat_id}::{msg_en}::{conv_sig}"
            cached_ai = _cache_get(resp_cache_key)
            computed_fresh = False
            if cached_ai:
                log_reasoning("Using cached AI response for this chat context.")
                ai_data = cached_ai
            else:
                computed_fresh = True
                # PERF: cheap local routing for common intents -> 1 Groq call instead of 2
                local_intent = None
                kb_keys = ["welfog_api"]
                txt = f" {msg_en} "
                comb = f"{original_msg} {msg_en}".lower()
                if any(w in txt for w in ["deal", "deals", "offer", "offers", "discount", "today deal"]):
                    local_intent = "deals"
                elif any(w in txt for w in ["category wise", "cat wise", "browse by category", "home page products", "category feed"]):
                    local_intent = "category_feed"
                elif any(w in txt for w in ["all categories", "categories list", "category list", "all category"]):
                    local_intent = "categories"
                elif _text_has_platform_overview_intent(comb):
                    local_intent = "general"
                    kb_keys = get_customer_kb_keys()
                elif _looks_like_conversational_followup(original_msg, msg_en) and len(recent_msgs) >= 2:
                    local_intent = "general"
                    kb_keys = get_customer_kb_keys()
                elif any(x in txt for x in ["staff", "department", "departments", "team", "teams"]):
                    local_intent = "general"
                    # For org/team questions, allow retrieval from all knowledge files.
                    kb_keys = get_customer_kb_keys()
                elif _text_has_delivery_or_order_area_intent(comb):
                    local_intent = "pincode_check"
                    kb_keys = ["shipping", "faqs", "company", "welfog_api"]
                elif _text_has_refund_or_return_intent(comb):
                    local_intent = "refund"
                    kb_keys += ["refund", "faqs"]
                elif _text_is_order_tracking_intent(comb):
                    local_intent = "order"
                    kb_keys += ["shipping", "faqs", "welfog_api"]
                elif any(w in txt for w in ["payment", "transaction", "upi"]):
                    local_intent = "payment"
                    kb_keys += ["payment", "faqs"]
                elif any(w in txt for w in ["seller", "become seller", "sell on welfog"]):
                    local_intent = "seller"
                    kb_keys += ["seller"]
                elif _text_has_product_shopping_intent(txt):
                    local_intent = "product"
                    kb_keys += ["faqs"]

                if local_intent:
                    log_reasoning(f"Local intent routing => {local_intent}")
                    # One-call answer with selected KB
                    kb_context = get_knowledge_context(
                        retrieval_query, keys=kb_keys, top_k=6, min_score=0.10
                    )
                    ai_data = ai_brain_answer(original_msg, kb_context, conv_for_llm) or {}
                    ai_data.setdefault("intent", local_intent)
                    ai_data.setdefault("is_welfog_related", True)
                    if local_intent == "order" and not _text_needs_order_id_for_tracking(comb):
                        ai_data["needs_order_id"] = False
                    if local_intent in ["refund", "payment"] and not _text_needs_order_id_for_refund_or_payment(comb):
                        ai_data["needs_order_id"] = False
                else:
                    log_reasoning("Using full AI router for intent + KB key selection.")
                    # Step 1: Route + choose knowledge files (Groq understands any language)
                    route_data = ai_brain_route(original_msg, conv_for_llm)
                    if not route_data:
                        fallback_text = sysmsg("server_busy")
                        return send_reply(fallback_text, lang)

                    kb_keys = route_data.get("kb_keys") or []
                    # Always include API playbook for shopping/deals/category flows
                    if route_data.get("intent") in ["product", "deals", "categories", "category_feed", "order"] and "welfog_api" not in kb_keys:
                        kb_keys = list(kb_keys) + ["welfog_api"]
                    # For non-shopping informational intents, don't ground on internal playbook files.
                    if route_data.get("intent") in ["general", "seller", "refund", "payment"] and kb_keys:
                        kb_keys = [k for k in kb_keys if k not in INTERNAL_KB_KEYS] or get_customer_kb_keys()
                    # If model didn't pick any KB keys for general info, search ALL customer KB (includes new admin files).
                    if route_data.get("intent") == "general":
                        kb_keys = get_customer_kb_keys()

                    # Step 2: Build KB context from selected files, then answer
                    kb_context = get_knowledge_context(
                        retrieval_query, keys=kb_keys, top_k=6, min_score=0.10
                    )
                    ai_data = ai_brain_answer(original_msg, kb_context, conv_for_llm)

                    if ai_data and "intent" not in ai_data:
                        # Safety: merge routing fields if model omitted them
                        ai_data["intent"] = route_data.get("intent", "general")
                        ai_data["is_welfog_related"] = route_data.get("is_welfog_related", True)

            # Note: ai_data already normalized above; avoid referencing route_data when local routing used.
            
            if not ai_data:
                grounded_fallback = direct_kb_search(retrieval_query, keys=get_customer_kb_keys(), min_score=0.34)
                if grounded_fallback:
                    log_reasoning("Groq unavailable; serving direct KB fallback.")
                    return send_reply(grounded_fallback, lang)
                fallback_text = sysmsg("server_busy")
                return send_reply(fallback_text, lang)

            apply_hinglish_product_fixes(original_msg, msg_en, ai_data)
            _merge_extracted_pincode(original_msg, msg_en, ai_data)
            if computed_fresh:
                _cache_set(resp_cache_key, ai_data)
            
            if ai_data.get("intent") == "order" and not _text_needs_order_id_for_tracking(comb_low):
                ai_data["needs_order_id"] = False
            if ai_data.get("intent") in ["refund", "payment"] and not _text_needs_order_id_for_refund_or_payment(comb_low):
                ai_data["needs_order_id"] = False
            intent = ai_data.get("intent", "general")
            is_welfog = ai_data.get("is_welfog_related", True)
            search_query = ai_data.get("search_query") or msg_en 
            ai_response_text = ai_data.get("response", "")

            # Strict anti-stale guard:
            # If the user asks a factual "who/owner/founder/partner" style query,
            # answer only when CURRENT KB has evidence. This prevents old memory answers
            # after file update/delete from admin panel.
            factual_query = _looks_like_factual_identity_query(f"{original_msg} {msg_en}")
            if intent in ["general", "seller", "refund", "payment"] and factual_query:
                kb_now_hit = best_kb_hit(retrieval_query, keys=get_customer_kb_keys(), min_score=0.31)
                if not kb_now_hit:
                    ai_response_text = (
                        "Is Welfog-related sawaal ke liye mere paas abhi confirmed official details "
                        "nahin milin. Kripya thoda detail se dobara likhen — jaise kis team, policy, ya feature ke baare mein puchhna hai."
                    )
                    log_reasoning("Welfog factual query: no supporting evidence in current KB snapshot.")

            # If search text itself looks like a category (including typos), switch to category-wise product fetch.
            if intent == "product":
                typed_cat_id = get_category_id_from_text(search_query, ctx=ctx)
                if typed_cat_id:
                    ctx.setdefault("data", {})
                    ctx["data"]["selected_category_id"] = typed_cat_id
                    search_query = ""
                    ai_data["search_query"] = ""

            # Domain Control for AI responses
            if not is_welfog or intent == "out_of_domain":
                reset_context(ctx)
                fallback_msg = ai_response_text if ai_response_text else sysmsg("out_of_domain")
                return send_reply(fallback_msg, lang)
        
# ================= INTENT EXECUTION =================
    if intent == "product":
        # If user selected a category previously, use it automatically
        selected_cat = ctx.get("data", {}).get("selected_category_id")
        selected_color = ctx.get("data", {}).get("selected_color")
        # Also detect color from current query (overrides previous)
        detected_color = _normalize_color(msg_en)
        if detected_color:
            selected_color = detected_color

        products = fetch_products_from_api(search_query, category_id=selected_cat, color=selected_color, page=1)
        if products:
            title_q = search_query.strip()
            if selected_cat and not title_q:
                response_text = sysmsg("products_title_category")
            else:
                response_text = sysmsg("products_title_query", query=search_query)
            
            # 🔥 FIX: Container me 'flex-direction: row' lagaya aur width set ki taaki horizontal scroll bane
            response_text += "<div class='wf-product-rail'>"

            for p in products:
                # Individual Card
                response_text += "<div class='wf-product-card'>"
                
                if p['image']:
                    response_text += f"<div style='width: 100%; height: 130px; background-color: #f9f9f9; border-radius: 8px; overflow: hidden; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; border: 1px solid #f0f0f0;'><img src='{p['image']}' alt='{p['name']}' style='max-width: 100%; max-height: 100%; object-fit: contain; display: block;'></div>"
                else:
                    response_text += f"<div style='width: 100%; height: 130px; background: #f0f0f0; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; color: #999; font-size: 12px; border: 1px solid #e0e0e0;'>{sysmsg('no_image')}</div>"
                
                name_short = p['name'][:38] + '...' if len(p['name']) > 38 else p['name']
                response_text += f"<div style='font-size: 13px; font-weight: 600; color: #333; margin-bottom: 6px; height: 34px; overflow: hidden; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;'>{name_short}</div>"
                response_text += f"<div style='font-size: 15px; font-weight: bold; color: #ff7a00; margin-bottom: 12px; margin-top: auto;'>₹{p['price']}</div>"
                
                if p['link']:
                    response_text += f"<a href='{p['link']}' target='_blank' rel='noopener noreferrer'>{sysmsg('view_product')}</a>"
                
                # 🔥 FIX: Yahan sirf Card wala div close hoga
                response_text += "</div>" 
            
            # 🔥 FIX: Loop khatam hone ke baad Horizontal Scroll wala div close hoga
            response_text += "</div>"
        else:
            if selected_cat and not (search_query or "").strip():
                response_text = sysmsg("category_products_not_found")
            else:
                response_text = sysmsg("product_not_found", query=search_query)
        reset_context(ctx)
    elif intent == "categories":
        cats = fetch_nav_categories()
        if not cats:
            response_text = sysmsg("categories_unavailable")
        else:
            # Try to extract a reasonable list from various possible shapes
            items = []
            if isinstance(cats, dict):
                for key in ["data", "categories", "result"]:
                    if isinstance(cats.get(key), list):
                        items = cats.get(key)
                        break
            elif isinstance(cats, list):
                items = cats

            # Flatten first-level items only
            shown = []
            for it in items[:20]:
                if not isinstance(it, dict):
                    continue
                cid = it.get("id") or it.get("category_id") or it.get("cat_id")
                name = it.get("name") or it.get("title") or it.get("category_name")
                if cid and name:
                    shown.append((cid, name))

            if not shown:
                response_text = sysmsg("categories_parse_failed")
            else:
                # Store categories map for next message selection
                ctx.setdefault("data", {})
                ctx["data"]["categories_map"] = {str(name).lower(): str(cid) for cid, name in shown}
                response_text = sysmsg("categories_title")
                response_text += sysmsg("categories_list_wrap_start")
                for cid, name in shown:
                    response_text += f"• <b>{name}</b> (id: {cid})<br>"
                response_text += sysmsg("categories_list_wrap_end") + sysmsg("categories_footer")
        # Keep context so next user message can select a category
        ctx["awaiting"] = "category_select"
    elif intent == "deals":
        deals = fetch_today_deals()
        items = []
        if isinstance(deals, dict):
            for key in ["data", "products", "result", "today_deal"]:
                if isinstance(deals.get(key), list):
                    items = deals.get(key)
                    break
        elif isinstance(deals, list):
            items = deals

        if not items:
            response_text = sysmsg("deals_unavailable")
        else:
            IMAGE_BASE_URL = "https://d1f02fefkbso7w.cloudfront.net/"
            response_text = sysmsg("deals_title", title=(deals.get("title") if isinstance(deals, dict) else None) or sysmsg("default_deals_name"))
            response_text += "<div class='wf-product-rail'>"

            shown = 0
            for p in items:
                if shown >= 5:
                    break
                if not isinstance(p, dict):
                    continue
                name = p.get("name") or p.get("product_name") or sysmsg("default_deal_card_title")
                new_price = p.get("new_price") or p.get("main_price") or p.get("price") or p.get("base_price") or sysmsg("na_price")
                old_price = p.get("old_price")
                slug = p.get("slug") or ""
                thumb = p.get("thumbnail_img") or p.get("thumbnail_image") or p.get("image") or ""
                image = (IMAGE_BASE_URL + str(thumb).lstrip("/")) if thumb else ""
                link = f"https://welfog.com/product/{slug}" if slug else "https://welfog.com"

                response_text += "<div class='wf-product-card'>"
                if image:
                    response_text += f"<div style='width: 100%; height: 130px; background-color: #f9f9f9; border-radius: 8px; overflow: hidden; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; border: 1px solid #f0f0f0;'><img src='{image}' alt='{name}' style='max-width: 100%; max-height: 100%; object-fit: contain; display: block;'></div>"
                else:
                    response_text += f"<div style='width: 100%; height: 130px; background: #f0f0f0; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; color: #999; font-size: 12px; border: 1px solid #e0e0e0;'>{sysmsg('no_image')}</div>"

                name_short = name[:38] + "..." if len(name) > 38 else name
                response_text += f"<div style='font-size: 13px; font-weight: 600; color: #333; margin-bottom: 6px; height: 34px; overflow: hidden; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;'>{name_short}</div>"
                if old_price and str(old_price).strip() and str(old_price) != str(new_price):
                    response_text += "<div style='margin-bottom: 10px; margin-top: auto;'>"
                    response_text += f"<span style='font-size: 15px; font-weight: bold; color: #ff7a00;'>₹{new_price}</span> "
                    response_text += f"<span style='font-size: 12px; color: #888; text-decoration: line-through;'>₹{old_price}</span>"
                    response_text += "</div>"
                else:
                    response_text += f"<div style='font-size: 15px; font-weight: bold; color: #ff7a00; margin-bottom: 12px; margin-top: auto;'>₹{new_price}</div>"
                response_text += f"<a href='{link}' target='_blank' rel='noopener noreferrer'>{sysmsg('view_deal')}</a>"
                response_text += "</div>"
                shown += 1

            response_text += "</div>"
        reset_context(ctx)
    elif intent == "category_feed":
        feed = fetch_category_wise_feed(page=1)
        groups = []
        if isinstance(feed, dict) and isinstance(feed.get("data"), list):
            groups = feed.get("data")

        if not groups:
            response_text = sysmsg("cat_feed_unavailable")
        else:
            IMAGE_BASE_URL = "https://d1f02fefkbso7w.cloudfront.net/"
            response_text = sysmsg("category_feed_title")
            shown_groups = 0
            for g in groups:
                if shown_groups >= 2:
                    break
                if not isinstance(g, dict):
                    continue
                cat = g.get("category") or {}
                cat_name = cat.get("name") or sysmsg("default_category_title")
                prods = g.get("products") if isinstance(g.get("products"), list) else []
                if not prods:
                    continue

                response_text += f"<div style='margin: 10px 0 6px 0; color:#333; font-weight:700;'>{cat_name}</div>"
                response_text += "<div class='wf-product-rail'>"
                shown = 0
                for p in prods:
                    if shown >= 5:
                        break
                    if not isinstance(p, dict):
                        continue
                    name = p.get("name") or sysmsg("default_product_card_title")
                    price = p.get("price") or sysmsg("na_price")
                    link_slug = p.get("link") or p.get("slug") or ""
                    thumb = p.get("image") or p.get("thumbnail_img") or p.get("thumbnail_image") or ""
                    image = (IMAGE_BASE_URL + str(thumb).lstrip("/")) if thumb else ""
                    link = f"https://welfog.com/product/{link_slug}" if link_slug else "https://welfog.com"

                    response_text += "<div class='wf-product-card'>"
                    if image:
                        response_text += f"<div style='width: 100%; height: 130px; background-color: #f9f9f9; border-radius: 8px; overflow: hidden; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; border: 1px solid #f0f0f0;'><img src='{image}' alt='{name}' style='max-width: 100%; max-height: 100%; object-fit: contain; display: block;'></div>"
                    else:
                        response_text += f"<div style='width: 100%; height: 130px; background: #f0f0f0; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; color: #999; font-size: 12px; border: 1px solid #e0e0e0;'>{sysmsg('no_image')}</div>"
                    name_short = name[:38] + "..." if len(name) > 38 else name
                    response_text += f"<div style='font-size: 13px; font-weight: 600; color: #333; margin-bottom: 6px; height: 34px; overflow: hidden; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;'>{name_short}</div>"
                    response_text += f"<div style='font-size: 15px; font-weight: bold; color: #ff7a00; margin-bottom: 12px; margin-top: auto;'>₹{price}</div>"
                    response_text += f"<a href='{link}' target='_blank' rel='noopener noreferrer'>{sysmsg('view_product')}</a>"
                    response_text += "</div>"
                    shown += 1

                response_text += "</div>"
                shown_groups += 1
        reset_context(ctx)
    elif intent == "pincode_check":
        pincode = ai_data.get("extracted_pincode", "")
        
        if not pincode:
            response_text = sysmsg("ask_pincode")
        else:
            api_res = check_pincode_delivery(pincode)
            
            # 🔥 Super Smart Response Logic
            if api_res and api_res.get("result") is True:
                message = api_res.get("message", "Product is available!")
                distance = api_res.get("distance", "nearby")
                
                response_text = (
                    f"✅ <b>Good News!</b><br>"
                    f"{message}<br><br>"
                    f"📍 Pincode: <b>{pincode}</b><br>"
                    f"🚚 Distance: <b>{distance}</b><br><br>"
                    f"you can place your order!"
                )
            elif api_res and api_res.get("result") is False:
                response_text = f"❌ sorry, PIN code <b>{pincode}</b> here we are not available to deliver , you can try another location ."
            else:
                # Agar API fail ho jaye ya response na mile
                response_text = sysmsg("server_technical_issue")
        
        reset_context(ctx)

    elif intent in ["order", "refund", "payment"]:
        ctx["last"] = intent
        needs_id = ai_data.get("needs_order_id", True) 
        current_order_id = ctx.get("order_id")

        # 🔥 MAJOR BUG FIX: Agar needs_id True hai (Tracking), tabhi ID maango.
        if needs_id:
            if not current_order_id:
                ctx["awaiting"] = "order_id"
                response_text = sysmsg("ask_order_id_for_intent", intent=intent)
            else:
                if intent == "order":
                    res = fetch_order_tracking(current_order_id)
                    response_text = (
                        format_order_tracking_reply(current_order_id, res)
                        if res
                        else "We could not find this order. Please check the Order ID from your confirmation email, SMS, or My Orders and try again."
                    )
                elif intent == "refund":
                    res = fetch_api("refund", current_order_id)
                    response_text = f"Refund status: {res['status']}. Expected in {res['time']}." if res else "No refund record found."
                elif intent == "payment":
                    res = fetch_api("payment", current_order_id)
                    response_text = f"Payment status: {res['status']} via {res['method']}." if res else "Payment details not found."
                reset_context(ctx)
        else:
            # FAQ-style order/refund/payment answer from model + KB (no order id required yet)
            response_text = ai_response_text if ai_response_text else sysmsg("how_can_i_help")
            if intent == "order":
                foot = sysmsg("order_tracking_optional_id_footer")
                if foot and response_text:
                    response_text = f"{response_text.rstrip()}<br><br>{foot}"
            reset_context(ctx)

    else:
        if ai_response_text:
            response_text = ai_response_text
        else:
            grounded = direct_kb_search(retrieval_query, keys=get_customer_kb_keys(), min_score=0.38)
            response_text = grounded if grounded else sysmsg("how_can_i_help_welfog")
        reset_context(ctx)

    # ================= FINAL RESPONSE =================
    return send_reply(response_text, lang)

def register_chat_routes(app):
    app.register_blueprint(chat_bp)
