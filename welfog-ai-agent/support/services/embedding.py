from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# ================= PERF: common embeddings & caches =================
GREETINGS = ["hi", "hello", "hii", "hey", "namaste"]
GREETINGS_VECS = model.encode(GREETINGS)
