# RAG Demo
A RAG pipeline built with ChromaDB, HuggingFace embeddings, Gemini, and FastAPI.

## Setup
1. Add PDFs to `./pdfs/`
2. Add `GEMINI_API_KEY` to `.env`
3. `pip install -r requirements.txt`
4. `python ingest.py`
5. `uvicorn main:app --reload`
