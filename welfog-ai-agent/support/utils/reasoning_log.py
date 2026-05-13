def log_reasoning(msg: str):
    """Uniform terminal reasoning logs for every chat path."""
    try:
        print(f"[AI Reasoning] {msg}")
    except Exception:
        pass
