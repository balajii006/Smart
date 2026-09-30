import os, re
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

load_dotenv()
for pkg in ("stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

STOP = set(stopwords.words("english"))
LEMMA = WordNetLemmatizer()
EMB = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensions

client = MongoClient(os.getenv("MONGO_URI"))
col = client[os.getenv("DB_NAME", "chatbot")][os.getenv("COLLECTION", "chunks")]
INDEX_NAME = "vector_index"


def clean(text: str) -> str:
    """NLTK cleaning: lowercase -> remove symbols -> drop stopwords -> lemmatize"""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(LEMMA.lemmatize(w) for w in words if w not in STOP)


def embed(text: str) -> list:
    return EMB.encode(text).tolist()
