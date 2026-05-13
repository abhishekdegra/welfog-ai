import requests

def ask_llama(prompt):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=5
        )
        return res.json().get("response")
    except:
        return None
    

def decide_action(message):
    msg = message.lower()

    # basic intent routing (temporary until full embedding system)
    if any(x in msg for x in ["order", "delivery", "track"]):
        return {"action": "get_orders", "query": message}

    if any(x in msg for x in ["buy", "search", "product", "show"]):
        return {"action": "search_products", "query": message}

    if any(x in msg for x in ["refund", "return"]):
        return {"action": "refund", "query": message}

    if any(x in msg for x in ["payment", "transaction"]):
        return {"action": "payment", "query": message}

    # fallback → AI
    return {"action": "ai", "query": message}


# import requests
# import json

# def ask_ai(message):
#     prompt = f"""
# You are an ecommerce customer support agent.
# Reply in max 2 lines. Be direct.

# User: {message}
# Agent:
# """

#     res = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "llama3",
#             "prompt": prompt,
#             "stream": False
#         }
#     )

#     return res.json()["response"]


# def decide_action(message):
#     prompt = f"""
# You are an ecommerce AI assistant.

# Decide action from:
# - search_products
# - get_orders
# - general

# User message: "{message}"

# Reply ONLY in JSON:
# {{"action": "...", "query": "..."}}
# """

#     res = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "llama3",
#             "prompt": prompt,
#             "stream": False
#         }
#     )

#     text = res.json()["response"]

#     try:
#         return json.loads(text)
#     except:
#         return {"action": "general", "query": message}