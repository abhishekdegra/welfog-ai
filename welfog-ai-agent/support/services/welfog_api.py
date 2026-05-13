import difflib
import re
from datetime import datetime, timedelta

import requests

from utils.helpers import _is_plausible_order_id

def fetch_api(endpoint, order_id):
    try:
        res = requests.get(f"http://localhost:5000/{endpoint}/{order_id}", timeout=10)
        return res.json() if res.status_code == 200 else None
    except:
        return None


TRACKING_URL = "https://welfogapi.welfog.com/api/onedelivery/welfog_track"
IMAGE_BASE = "https://d1f02fefkbso7w.cloudfront.net/"


def fetch_order_tracking(order_id):
    """
    Live order status from OneDelivery tracking API.
    POST JSON body: {"orderId": "<digits>"} — orderId must be a string in JSON.
    """
    if not order_id:
        return None
    oid = str(order_id).strip()
    if not _is_plausible_order_id(oid):
        return None
    try:
        res = requests.post(TRACKING_URL, json={"orderId": oid}, timeout=15)
        if res.status_code != 200:
            return None
        data = res.json()
        if not isinstance(data, dict) or data.get("result") != "ok":
            return None
        return data
    except Exception as e:
        print(f"Order tracking API error: {e}")
        return None


def _eta_from_order_date_and_minutes(order_date_val, minutes_val):
    """
    expected_delivery from API = minutes from start of order_date (date-only, local midnight).
    Returns (formatted_datetime_str, days_equivalent_float) or (None, None).
    """
    if not order_date_val or minutes_val is None or minutes_val == "":
        return None, None
    try:
        minutes = int(float(minutes_val))
    except (TypeError, ValueError):
        return None, None
    if minutes < 0:
        return None, None
    raw = str(order_date_val).strip()
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", raw)
    if not m:
        return None, None
    try:
        day = datetime.strptime(m.group(1), "%Y-%m-%d")
    except ValueError:
        return None, None
    eta = day + timedelta(minutes=minutes)
    days_equiv = round(minutes / 1440.0, 1)
    # e.g. "15 May 2026, 00:56" — day-first, English month (no locale dependency)
    label = f"{eta.day} {eta.strftime('%B %Y')}, {eta.strftime('%H:%M')}"
    return label, days_equiv


def format_order_tracking_reply(order_id, data):
    """Build HTML user message from welfog_track API payload (result ok)."""
    title = (data.get("product_title") or "").strip() or "Order"
    status_human = (data.get("current_order_status") or "—").strip()
    order_date = data.get("order_date")
    pay = data.get("payment_status")
    ptype = data.get("payment_type")
    ed = data.get("expected_delivery")
    tc = data.get("tracking_code")
    img = (data.get("product_img") or "").strip()

    lines = [f"<b>Order ID:</b> {order_id}"]
    if img:
        src = IMAGE_BASE + img.lstrip("/")
        lines.append(
            "<div style='width:100%;max-width:360px;height:130px;background:#f9f9f9;border-radius:8px;"
            "overflow:hidden;margin:10px 0;display:flex;align-items:center;justify-content:center;"
            "border:1px solid #f0f0f0;'>"
            f"<img src='{src}' alt='' style='max-width:100%;max-height:100%;object-fit:contain;'/>"
            "</div>"
        )
    lines.append(f"<b>{title}</b>")
    lines.append(f"<b>Current status:</b> {status_human}")
    if order_date:
        lines.append(f"<b>Order date:</b> {order_date}")
    eta_label, days_equiv = _eta_from_order_date_and_minutes(order_date, ed)
    if eta_label is not None:
        extra = f" (~{days_equiv} days from order date)" if days_equiv is not None else ""
        lines.append(f"<b>Expected delivery:</b> {eta_label}{extra}")
    if pay:
        ptype_disp = ptype.replace("_", " ") if isinstance(ptype, str) else ptype
        lines.append(f"<b>Payment:</b> {pay}" + (f" ({ptype_disp})" if ptype_disp else ""))
    if tc:
        lines.append(f"<b>Tracking code:</b> {tc}")
    return "<br>".join(lines)


def _normalize_color(text: str):
    if not text:
        return None
    t = text.lower()
    # common colors (extend as needed)
    if "multicolor" in t or "multi color" in t:
        return "Multicolor"
    if "black" in t:
        return "Black"
    if "white" in t:
        return "White"
    if "red" in t:
        return "Red"
    if "blue" in t:
        return "Blue"
    if "green" in t:
        return "Green"
    if "yellow" in t:
        return "Yellow"
    if "pink" in t:
        return "Pink"
    if "purple" in t:
        return "Purple"
    if "brown" in t:
        return "Brown"
    if "grey" in t or "gray" in t:
        return "Grey"
    return None

# ================= EXTERNAL APIs =================
def fetch_products_from_api(query, category_id=None, color=None, page=1):
    from services import kb_service as _kb
    sysmsg = _kb.sysmsg
    # Allow category-only browse (query can be empty if category_id is provided)
    if (not query or not query.strip()) and not category_id:
        return []
        
    query = (query or "").lower()
    # 1. CLEAN QUERY (LLM agar kuch chhod de toh yahan saaf karo)
    stop_words = ["all", "show", "me", "the", "a", "an", "buy", "need", "want", "dikha", "chahiye", "saare", "mere", "ko", "bhai"]
    q_words = [w for w in query.split() if w not in stop_words]
    clean_query = " ".join(q_words)
    if not clean_query: clean_query = query

    # 2. 🔥 THE "ANTI-CATEGORY" LOGIC (Super Smart Filter)
    anti_words = []
    # Agar user ne 'cover' ya 'case' nahi manga, toh cover mat dikhao!
    if "cover" not in clean_query and "case" not in clean_query and "bumper" not in clean_query:
        anti_words.extend(["cover", "case", "bumper", "glass", "protector"])
    # Agar user ne charger/cable nahi manga, toh wo mat dikhao!
    if "charger" not in clean_query and "cable" not in clean_query and "adapter" not in clean_query:
        anti_words.extend(["charger", "cable", "adapter", "usb"])

    brands = ["samsung", "iphone", "apple", "vivo", "oppo", "realme", "mi", "xiaomi", "oneplus", "durex"]
    query_brands = [w for w in q_words if w in brands]

    try:
        url = "https://welfogapi.welfog.com/api/v2/products/search"
        params = {"page": page or 1, "latitude": "", "longitude": ""}
        if clean_query.strip():
            params["name"] = clean_query
        if category_id:
            params["categories"] = str(category_id)
        if color:
            params["color"] = color

        res = requests.get(url, params=params, timeout=10)
        data = res.json() if res.status_code == 200 else {}
        products_list = data.get("data", [])
        
        # FALLBACK SEARCH
        if not products_list and len(q_words) > 1 and not category_id:
            fallback_word = q_words[0] if any(b in q_words[0] for b in brands) else q_words[-1]
            res = requests.get(url, params={"page": 1, "name": fallback_word, "latitude": "", "longitude": ""}, timeout=10)
            data = res.json() if res.status_code == 200 else {}
            products_list = data.get("data", [])

        if not products_list: 
            return []

        IMAGE_BASE_URL = "https://d1f02fefkbso7w.cloudfront.net/"
        scored_products = []

        for p in products_list:
            name = p.get("name") or sysmsg("default_product_card_title")
            name_lower = name.lower()
            
            # 🔥 STRICT BRAND CHECK
            if query_brands and not any(qb in name_lower for qb in query_brands):
                continue
                
            # 🔥 STRICT ANTI-WORD CHECK (Mobile manga toh Cover reject)
            # Dhyan rahe exact word match ho, naam ka hissa nahi
            product_words = name_lower.split()
            if any(aw in product_words for aw in anti_words):
                continue

            # SCORING SYSTEM
            score = 0
            for word in q_words:
                singular = word[:-1] if word.endswith('s') else word
                if word in name_lower or singular in name_lower:
                    score += 2
                elif len(word) >= 4:
                    for n_word in product_words:
                        if abs(len(word) - len(n_word)) <= 1:
                            match_count = sum(1 for a, b in zip(word, n_word) if a == b)
                            if match_count / len(word) >= 0.75:
                                score += 1
                                break
            
            if score > 0 or not q_words:
                scored_products.append({
                    "name": name,
                    "price": p.get("main_price") or sysmsg("na_price"),
                    "image": (IMAGE_BASE_URL + p.get("thumbnail_image", "").lstrip('/')) if p.get("thumbnail_image") else "",
                    "link": f"https://welfog.com/product/{p.get('slug', '')}" if p.get("slug") else "https://welfog.com",
                    "score": score
                })

        scored_products.sort(key=lambda x: x['score'], reverse=True)
        return scored_products[:5]

    except Exception as e:
        print(f"Product Fetch Error: {e}")
        return []    


def check_pincode_delivery(pincode):
    try:
        url = "https://welfogapi.welfog.com/api/v2/pincode/check_pincode"
        # Pincode ko hamesha string mein convert karke bhejo
        payload = {'pincode': str(pincode)} 
        
        # API hit
        res = requests.post(url, data=payload, timeout=10)
        
        if res.status_code == 200:
            return res.json()
        else:
            print(f"⚠️ API Status Error: {res.status_code}")
            return None
    except Exception as e:
        print(f"❌ Pincode API Exception: {e}")
        return None


def fetch_nav_categories():
    try:
        url = "https://welfogapi.welfog.com/api/nav_cat_data"
        res = requests.get(url, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception as e:
        print(f"❌ Categories API Exception: {e}")
        return None


# -------- Category name -> id resolver (cached) --------
_navcat_cache = {"ts": 0.0, "map": {}}
_navcat_ttl_sec = 600  # 10 minutes


def _normalize_cat_name(s: str):
    if not s:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_category_id_from_text(text: str, ctx=None):
    """
    Returns category_id (string) if any category name matches the text.
    Uses ctx.categories_map first, else cached nav_cat_data map.
    """
    t = _normalize_cat_name(text)
    if not t:
        return None

    def _fuzzy_lookup(input_text: str, mapping: dict):
        if not mapping:
            return None
        # exact / contains match
        for name, cid in mapping.items():
            if name and name in input_text:
                return str(cid)
        # token-level typo match (e.g. electonics -> electronics)
        words = [w for w in input_text.split() if len(w) >= 4]
        names = list(mapping.keys())
        for w in words:
            best = difflib.get_close_matches(w, names, n=1, cutoff=0.82)
            if best:
                return str(mapping[best[0]])
        # whole phrase fuzzy
        phrase = difflib.get_close_matches(input_text, names, n=1, cutoff=0.75)
        if phrase:
            return str(mapping[phrase[0]])
        return None

    # 1) from ctx map (fastest)
    if ctx:
        cat_map = (ctx.get("data") or {}).get("categories_map") or {}
        if cat_map:
            normalized_map = {}
            for name, cid in cat_map.items():
                nn = _normalize_cat_name(name)
                if nn:
                    normalized_map[nn] = str(cid)
            hit = _fuzzy_lookup(t, normalized_map)
            if hit:
                return hit

    # 2) from global cache
    now_ts = datetime.now().timestamp()
    cached_map = _navcat_cache.get("map") or {}
    if cached_map and (now_ts - float(_navcat_cache.get("ts") or 0)) < _navcat_ttl_sec:
        hit = _fuzzy_lookup(t, cached_map)
        if hit:
            return hit
        return None

    # 3) refresh cache from API
    cats = fetch_nav_categories()
    items = []
    if isinstance(cats, dict):
        for key in ["data", "categories", "result"]:
            if isinstance(cats.get(key), list):
                items = cats.get(key)
                break
    elif isinstance(cats, list):
        items = cats

    new_map = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = it.get("id") or it.get("category_id") or it.get("cat_id")
        name = it.get("name") or it.get("title") or it.get("category_name")
        nn = _normalize_cat_name(name)
        if cid and nn:
            new_map[nn] = str(cid)

    _navcat_cache["ts"] = now_ts
    _navcat_cache["map"] = new_map

    hit = _fuzzy_lookup(t, new_map)
    if hit:
        return hit
    return None


def fetch_today_deals(latitude="34.04505157470703", longitude="78.38422393798828"):
    try:
        url = "https://welfogapi.welfog.com/api/today_deal"
        res = requests.get(url, params={"latitude": latitude, "longitude": longitude}, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception as e:
        print(f"❌ Deals API Exception: {e}")
        return None


def fetch_category_wise_feed(page=1, latitude="34.04505157470703", longitude="78.38422393798828"):
    try:
        url = "https://welfogapi.welfog.com/api/cat_wise_product_show"
        res = requests.get(url, params={"latitude": latitude, "longitude": longitude, "page": page}, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception as e:
        print(f"❌ Cat-wise API Exception: {e}")
        return None
