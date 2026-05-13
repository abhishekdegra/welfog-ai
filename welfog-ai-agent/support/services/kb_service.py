import os
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from support_paths import KNOWLEDGE_DIR
from services.embedding import model
from utils.cache import _cache, _cache_get, _cache_set
from utils.reasoning_log import log_reasoning
from utils import validators
from utils.helpers import (
    _is_plausible_order_id,
    _looks_like_greeting_message,
    _text_has_refund_or_return_intent,
    _text_is_order_id_help_request,
    _text_is_order_tracking_intent,
    _text_needs_order_id_for_refund_or_payment,
    _text_needs_order_id_for_tracking,
)

_KB_SNAPSHOT = ""
def get_runtime_knowledge_files():
    """
    Auto-discovers all .txt files from knowledge folder.
    No hardcoded mapping required.
    """
    runtime = {}
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        return runtime
    for filename in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(KNOWLEDGE_DIR, filename)
        if not os.path.isfile(file_path):
            continue
        key_base = os.path.splitext(filename)[0].replace("-", "_").replace(" ", "_").lower()
        key = re.sub(r"[^a-z0-9_]", "", key_base)
        if not key:
            continue
        # Avoid key collisions if two files normalize to same key
        if key in runtime:
            n = 2
            while f"{key}_{n}" in runtime:
                n += 1
            key = f"{key}_{n}"
        runtime[key] = file_path
    return runtime

def get_allowed_knowledge_filenames():
    return sorted({os.path.basename(path) for path in get_runtime_knowledge_files().values() if path.endswith(".txt")})

def _compute_kb_snapshot(runtime_files):
    parts = []
    for key, path in sorted(runtime_files.items()):
        try:
            st = os.stat(path)
            parts.append(f"{key}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{key}:missing")
    return "|".join(parts)


def _parse_system_messages(text: str):
    """
    Parses lines like: key = value
    Ignores empty lines and headings.
    """
    out = {}
    if not text:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            out[k] = v
    return out


def _split_kb_chunks(text: str):
    """
    Split knowledge files into embedding chunks. Short title-only blocks are merged
    with the following paragraph so retrieval returns real content, not just headings.
    """
    if not text:
        return []
    parts = []
    for c in text.split("\n\n"):
        c = c.strip().replace("\n", "<br>")
        if len(c) < 12:
            continue
        parts.append(c)
    if not parts:
        return []
    merged = []
    acc = parts[0]
    for p in parts[1:]:
        if len(acc) < 160:
            acc = acc + "<br><br>" + p
        else:
            merged.append(acc)
            acc = p
    merged.append(acc)
    out = []
    max_len, step = 950, 780
    for ch in merged:
        if len(ch) <= max_len:
            out.append(ch)
            continue
        i = 0
        while i < len(ch):
            piece = ch[i : i + max_len]
            if len(piece) >= 20:
                out.append(piece)
            i += step
    return out


def load_knowledge_index(runtime_files=None):
    chunks_by_key = {}
    vectors_by_key = {}
    all_chunks_local = []
    all_sources_local = []

    files_map = runtime_files if runtime_files is not None else get_runtime_knowledge_files()
    for k, path in files_map.items():
        try:
            if not os.path.exists(path):
                print(f"⚠️ Warning: File missing at {path}")
                chunks_by_key[k] = []
                vectors_by_key[k] = []
                continue

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = _split_kb_chunks(content)
            chunks_by_key[k] = chunks
            if chunks:
                vecs = model.encode(chunks)
                vectors_by_key[k] = vecs
                all_chunks_local.extend(chunks)
                all_sources_local.extend([k] * len(chunks))
            else:
                vectors_by_key[k] = []
        except Exception as e:
            print(f"Error loading {path}: {e}")
            chunks_by_key[k] = []
            vectors_by_key[k] = []

    all_vectors_local = model.encode(all_chunks_local) if all_chunks_local else []
    return chunks_by_key, vectors_by_key, all_chunks_local, all_vectors_local, all_sources_local

def refresh_knowledge_cache():
    global kb_chunks_by_key, kb_vectors_by_key, all_chunks, all_vectors, all_chunk_sources, _SYSTEM_MESSAGES, _KB_SNAPSHOT
    runtime_files = get_runtime_knowledge_files()
    kb_chunks_by_key, kb_vectors_by_key, all_chunks, all_vectors, all_chunk_sources = load_knowledge_index(runtime_files)
    _KB_SNAPSHOT = _compute_kb_snapshot(runtime_files)
    try:
        sys_path = runtime_files.get("system_messages")
        if sys_path and os.path.exists(sys_path):
            with open(sys_path, "r", encoding="utf-8") as f:
                _SYSTEM_MESSAGES = _parse_system_messages(f.read())
        else:
            _SYSTEM_MESSAGES = {}
    except Exception as e:
        print("System messages reload error:", e)

def ensure_knowledge_cache_fresh():
    global _KB_SNAPSHOT
    runtime_files = get_runtime_knowledge_files()
    latest_snapshot = _compute_kb_snapshot(runtime_files)
    if latest_snapshot != _KB_SNAPSHOT:
        refresh_knowledge_cache()
        _cache.clear()


_SYSTEM_MESSAGES = {}
refresh_knowledge_cache()

INTERNAL_KB_KEYS = {"welfog_api", "system_messages"}


def get_customer_kb_keys():
    return [k for k in get_runtime_knowledge_files().keys() if k not in INTERNAL_KB_KEYS]


def sysmsg(key: str, **kwargs):
    """
    Fetch a user-facing message from knowledge files.
    Supports {placeholders}.
    """
    txt = _SYSTEM_MESSAGES.get(key, "")
    if not txt:
        return ""
    try:
        return txt.format(**kwargs)
    except Exception:
        return txt



def get_knowledge_context(query, keys=None, top_k=3, min_score=0.15):
    """
    If keys provided, search within those knowledge files only; else search all.
    Returns an HTML string to inject into the system prompt.
    """
    if not all_chunks:
        return ""

    # Cache KB context because same query repeats often (especially with translations)
    cache_key = f"kbctx::{query}::{','.join(keys) if keys else 'ALL'}::{top_k}::{min_score}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if keys:
        keys = [k for k in keys if k in kb_chunks_by_key]
        scoped_chunks = []
        scoped_vectors_list = []
        scoped_sources = []
        for k in keys:
            ch = kb_chunks_by_key.get(k)
            if not ch:
                continue
            vec = kb_vectors_by_key.get(k)
            if vec is None:
                continue
            if len(ch) == len(vec):
                scoped_chunks.extend(ch)
                scoped_vectors_list.append(vec)
                scoped_sources.extend([k] * len(ch))
        if not scoped_chunks:
            _cache_set(cache_key, "")
            return ""

        # concat precomputed vectors (fast)
        import numpy as np
        scoped_vectors = np.vstack(scoped_vectors_list) if len(scoped_vectors_list) > 1 else scoped_vectors_list[0]

        query_vec = model.encode([query])
        scores = cosine_similarity(query_vec, scoped_vectors)[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        picked = [(scoped_sources[i], scoped_chunks[i], float(scores[i])) for i in top_indices if scores[i] > min_score]
    else:
        query_vec = model.encode([query])
        scores = cosine_similarity(query_vec, all_vectors)[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        picked = [(all_chunk_sources[i], all_chunks[i], float(scores[i])) for i in top_indices if scores[i] > min_score]

    if not picked:
        return ""

    # Include sources so the model knows which file the context came from.
    out = []
    for src, chunk, sc in picked:
        out.append(f"[source={src} score={sc:.2f}] {chunk}")
    result = "<br><br>".join(out)
    _cache_set(cache_key, result)
    return result


# ================= 🚀 1.5 DIRECT KB SEARCH 🚀 =================
def best_kb_hit(query, keys=None, min_score=0.28):
    """
    Returns the best matching KB chunk (raw HTML chunk) with its score and source.
    Use this to prefer KB-first answers without hardcoding filenames.
    """
    if not all_chunks:
        return None
    if not query:
        return None

    if keys:
        keys = [k for k in keys if k in kb_chunks_by_key]
        scoped_chunks = []
        scoped_vectors_list = []
        scoped_sources = []
        for k in keys:
            ch = kb_chunks_by_key.get(k) or []
            vec = kb_vectors_by_key.get(k)
            if ch and vec is not None and len(ch) == len(vec):
                scoped_chunks.extend(ch)
                scoped_vectors_list.append(vec)
                scoped_sources.extend([k] * len(ch))
        if not scoped_chunks:
            return None
        import numpy as np
        scoped_vectors = np.vstack(scoped_vectors_list) if len(scoped_vectors_list) > 1 else scoped_vectors_list[0]
        query_vec = model.encode([query])
        scores = cosine_similarity(query_vec, scoped_vectors)[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        if best_score >= min_score:
            return {"source": scoped_sources[best_idx], "chunk": scoped_chunks[best_idx], "score": best_score}
        return None
    else:
        query_vec = model.encode([query])
        scores = cosine_similarity(query_vec, all_vectors)[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        if best_score >= min_score:
            return {"source": all_chunk_sources[best_idx], "chunk": all_chunks[best_idx], "score": best_score}
        return None


def keyword_kb_hit(query: str, keys=None, min_hits: int = 2):
    """
    Fallback when embeddings miss due to phrasing/short queries.
    Scores chunks by keyword overlap (case-insensitive substring match).
    Returns: {source, chunk, score} where score is hit-count.
    """
    if not query:
        return None

    # tokenize: keep alphanum, split, drop short words
    q = re.sub(r"[^a-z0-9 ]+", " ", (query or "").lower())
    raw_tokens = [t for t in q.split() if len(t) >= 4]
    if not raw_tokens:
        return None

    stop = {
        "welfog",
        "about",
        "explain",
        "please",
        "tell",
        "criteria",
        "eligibility",  # keep? removing avoids overfitting on header-only; other tokens still match
        "rules",
    }
    tokens = [t for t in raw_tokens if t not in stop]
    if not tokens:
        tokens = raw_tokens

    if keys:
        keys = [k for k in keys if k in kb_chunks_by_key]
        scoped = []
        scoped_src = []
        for k in keys:
            ch = kb_chunks_by_key.get(k) or []
            scoped.extend(ch)
            scoped_src.extend([k] * len(ch))
    else:
        scoped = all_chunks or []
        scoped_src = all_chunk_sources or []

    best = None
    best_hits = 0
    for i, chunk in enumerate(scoped):
        if not chunk:
            continue
        low = chunk.lower()
        hits = sum(1 for t in tokens if t in low)
        if hits > best_hits:
            best_hits = hits
            best = (scoped_src[i], chunk)
            # early exit if very strong
            if best_hits >= max(min_hits + 2, 5):
                break

    if best and best_hits >= min_hits:
        src, ch = best
        return {"source": src, "chunk": ch, "score": float(best_hits)}
    return None


def direct_kb_search(query, keys=None, min_score=0.40):
    if not all_chunks:
        return None
    picked = None
    if keys:
        keys = [k for k in keys if k in kb_chunks_by_key]
        scoped_chunks = []
        scoped_vectors_list = []
        for k in keys:
            ch = kb_chunks_by_key.get(k) or []
            vec = kb_vectors_by_key.get(k)
            if ch and vec is not None and len(ch) == len(vec):
                scoped_chunks.extend(ch)
                scoped_vectors_list.append(vec)
        if not scoped_chunks:
            return None
        import numpy as np
        scoped_vectors = np.vstack(scoped_vectors_list) if len(scoped_vectors_list) > 1 else scoped_vectors_list[0]
        query_vec = model.encode([query])
        scores = cosine_similarity(query_vec, scoped_vectors)[0]
        best_idx = scores.argmax()
        if scores[best_idx] > min_score:
            picked = scoped_chunks[best_idx]
    else:
        query_vec = model.encode([query])
        scores = cosine_similarity(query_vec, all_vectors)[0]
        best_idx = scores.argmax()
        if scores[best_idx] > min_score:
            picked = all_chunks[best_idx]

    if picked:
        cleaned = re.sub(r"\[source=.*?score=.*?\]\s*", "", picked, flags=re.IGNORECASE)
        return f"According to our knowledge base:<br><b>{cleaned}</b>"
    return None


# ================= ⚡ ULTRA-FAST INSTANT ROUTER (Dumb & Fast) ⚡ =================
def smart_instant_router(original_msg, english_msg):
    text = f" {original_msg} {english_msg} ".lower()
    words = original_msg.lower().split()

    # 1. INSTANT GREETINGS CHECK (Static)
    greetings = ["hi", "hello","hii","heyy", "hey", "namaste", "bhai", "sun", "suno", "brother", "hiii"]
    if _looks_like_greeting_message(original_msg) or (len(words) <= 3 and all(w in greetings for w in words)):
        return {"action": "text", "data": sysmsg("greeting")}

    # 2. INSTANT REJECTION & OFF-TOPIC (Static Guardrails)
    competitors = ["amazon", "flipkart", "myntra", "meesho", "ajio", "snapdeal", "shopsy", "glowroad", "zomato", "swiggy", "groww"]
    off_topics = [
        "school", "college", "university", "exam", "homework", "free fire", "pubg", "bgmi", "game", "movie", "song",
        "cricket", "football", "ipl", "weather", "politics", "election", "bollywood", "astrology", "kundli",
        "girlfriend", "boyfriend", "breakup", "joke", "funny", "recipe", "cooking",
    ]

    if any(f" {c} " in text for c in competitors):
        return {"action": "reject", "data": "I’m here to assist you with Welfog 😊\nI don't have information about other companies."}

    if any(f" {ot} " in text for ot in off_topics):
        polite = sysmsg("off_topic_polite")
        return {"action": "reject", "data": polite or "I'm a Welfog assistant and can only help with Welfog shopping and support."}

    # 3. ORDER ID / TRACKING HELP SHORT-CIRCUIT
    if _text_is_order_id_help_request(text):
        return {"action": "text", "data": sysmsg("order_id_help")}
    if _text_has_refund_or_return_intent(text) and not _text_needs_order_id_for_refund_or_payment(text):
        return {"action": "text", "data": sysmsg("refund_payment_help")}
    if _text_is_order_tracking_intent(text) and not _text_needs_order_id_for_tracking(text):
        return {"action": "text", "data": sysmsg("tracking_help")}

    # 4. DIRECT ORDER ID ENTRY (Regex - 100% accurate; 6-digit PIN is not an order id)
    if len(words) == 1 and _is_plausible_order_id(words[0]):
        w0 = words[0].lower()
        if re.fullmatch(r"[1-9]\d{5}", w0):
            return None
        return {"action": "direct_order_id", "order_id": words[0].upper()}

    # 5. INSTANT SOCIAL MEDIA LINKS
    is_social = any(w in text for w in ["instagram", "insta", "linkedin", "facebook", "youtube", "twitter"])
    is_welfog_context = any(w in text for w in ["welfog", "company", "official", "our", "your", "apna"])
    if is_social:
        if not is_welfog_context and len(words) > 3:
            return {"action": "reject", "data": "I can only provide official social media links for Welfog. 😊"}
        else:
            links = []
            btn_style = "display: block; text-align: center; max-width: 220px; margin-bottom: 12px; color: white; padding: 10px 15px; text-decoration: none; border-radius: 25px; font-weight: bold; font-size: 14px; box-shadow: 0 4px 6px rgba(0,0,0,0.15);"
            if "instagram" in text or "insta" in text:
                links.append(f"<a href='https://www.instagram.com/welfog_online/' target='_blank' style='{btn_style} background: linear-gradient(45deg, #f09433 0%, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%);'>📸 Instagram</a>")
            if "twitter" in text or " x " in text:
                links.append(f"<a href='https://x.com/welfog' target='_blank' style='{btn_style} background: linear-gradient(to right, #14171A, #000000);'>🐦 Twitter (X)</a>")
            if "facebook" in text or " fb " in text:
                links.append(f"<a href='https://www.facebook.com/people/welfog/' target='_blank' style='{btn_style} background: linear-gradient(to right, #1877F2, #0b50a8);'>📘 Facebook</a>")
            if "youtube" in text:
                links.append(f"<a href='https://www.youtube.com/@welfog_online' target='_blank' style='{btn_style} background: linear-gradient(to right, #FF0000, #b30000);'>▶️ YouTube</a>")
            if "linkedin" in text or "linkdin" in text:
                links.append(f"<a href='https://www.linkedin.com/company/welfog/' target='_blank' style='{btn_style} background: linear-gradient(to right, #0077B5, #005582);'>💼 LinkedIn</a>")
            if links:
                return {"action": "text", "data": "<div style='font-size: 14px; color: #333; margin-bottom: 12px;'><b>Here are our official links:</b></div>" + "".join(links)}
    # 🔥 SAB KUCH HATA DIYA! Koi keyword matching nahi. 
    # Ab chahe Telugu me puche ya Hindi me, direct AI Brain decide karega!
    return None
