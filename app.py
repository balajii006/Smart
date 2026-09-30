"""Smart Chat Assistant - RAG chatbot (run: python app.py)"""
import os
import ollama
from common import col, clean, embed, INDEX_NAME

MODEL = os.getenv("LLAMA_MODEL", "llama3.2")
THRESHOLD = 0.60   # below this score -> question is "unrelated" -> general query
TOP_K = 3


def retrieve(question):
    q_vec = embed(clean(question))
    pipeline = [
        {"$vectorSearch": {"index": INDEX_NAME, "path": "embedding",
                           "queryVector": q_vec, "numCandidates": 100, "limit": TOP_K}},
        {"$project": {"_id": 0, "text": 1,
                      "score": {"$meta": "vectorSearchScore"}}},
    ]
    return [r for r in col.aggregate(pipeline) if r["score"] >= THRESHOLD]


def ask(question, history):
    chunks = retrieve(question)
    if chunks:   # RAG path
        context = "\n\n".join(c["text"] for c in chunks)
        prompt = (f"Answer using ONLY the context below. If the answer is not "
                  f"in it, say you don't know.\n\nContext:\n{context}\n\n"
                  f"Question: {question}")
        mode = f"RAG ({len(chunks)} chunks)"
    else:        # general path
        prompt, mode = question, "General"

    messages = history + [{"role": "user", "content": prompt}]
    reply = ollama.chat(model=MODEL, messages=messages)["message"]["content"]
    history += [{"role": "user", "content": question},
                {"role": "assistant", "content": reply}]
    return reply, mode


if __name__ == "__main__":
    history = []
    print("Smart Chat Assistant ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if q:
            answer, mode = ask(q, history)
            print(f"\nBot [{mode}]: {answer}\n")
