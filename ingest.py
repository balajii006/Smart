"""Run once: reads data/*.txt -> chunks -> embeddings -> MongoDB Atlas + vector index"""
import glob
from pymongo.operations import SearchIndexModel
from common import col, clean, embed, INDEX_NAME


def chunk(text, size=60, overlap=15):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


col.delete_many({})
docs = []
for path in glob.glob("data/*.txt"):
    for c in chunk(open(path, encoding="utf-8").read()):
        docs.append({"text": c, "source": path, "embedding": embed(clean(c))})

col.insert_many(docs)
print(f"Inserted {len(docs)} chunks")

try:
    col.create_search_index(SearchIndexModel(
        name=INDEX_NAME,
        type="vectorSearch",
        definition={"fields": [{
            "type": "vector", "path": "embedding",
            "numDimensions": 384, "similarity": "cosine"}]},
    ))
    print("Vector index created (wait ~1 min until it is READY in Atlas)")
except Exception as e:
    print("Index note:", e)
