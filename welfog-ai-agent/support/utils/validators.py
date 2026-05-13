import re


def _safe_knowledge_filename(name: str) -> str:
    if not name:
        return ""
    cleaned = name.strip().lower().replace(" ", "_")
    cleaned = re.sub(r"[^a-z0-9_-]", "", cleaned)
    if not cleaned or len(cleaned) > 64:
        return ""
    return cleaned
