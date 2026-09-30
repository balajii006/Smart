# Smart Chat Assistant - RAG-based Conversational AI

**Tech:** Python | NLTK | MongoDB Atlas Vector Search | Llama (Ollama)

## About
A RAG chatbot that cleans user questions with NLTK, converts them into
embeddings, and finds relevant chunks using MongoDB Atlas Vector Search.
Retrieved context is added to the prompt so Llama gives accurate,
context-based answers. Unrelated questions go straight to Llama as
general queries.

## Flow
```
Question -> NLTK clean -> Embedding -> MongoDB Vector Search
                                          |
                     score >= 0.60 ? -----+----- No -> Llama (general query)
                          | Yes
            Add chunks to prompt -> Llama -> Context-based answer
```

## Project Structure
```
smart-chat-assistant/
├── common.py         # NLTK cleaning, embeddings, MongoDB connection
├── ingest.py         # chunk docs -> embed -> store + create vector index
├── app.py            # chat loop (RAG / general routing + Llama)
├── data/sample.txt   # your documents
├── requirements.txt
└── .env.example
```

## Setup
1. Install packages:
   `pip install -r requirements.txt`
2. Install Ollama (https://ollama.com), then:
   `ollama pull llama3.2`
3. Create a free MongoDB Atlas cluster and copy the connection string.
4. Copy `.env.example` to `.env` and paste your `MONGO_URI`.
5. Put your `.txt` documents inside `data/`.
6. Load data and create the index:
   `python ingest.py`
   (wait ~1 min until the vector index is READY in Atlas)
7. Start the chatbot:
   `python app.py`

## Example Questions
- "What is the grade for 92 percent?"  -> RAG answer (from documents)
- "What is Python?"                    -> General answer (from Llama)

## Notes
- Embedding model: all-MiniLM-L6-v2 (384 dimensions, cosine similarity)
- Similarity threshold: 0.60 (change `THRESHOLD` in app.py)
- Top chunks retrieved: 3 (change `TOP_K` in app.py)
