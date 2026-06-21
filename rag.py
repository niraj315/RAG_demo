import os 
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

#Config
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_docs"
EMBED_MODEL ="all-MiniLM-L6-v2"
TOP_K = 4 #number of chunks to retrive
GEMINI_MODEL = "gemini-3-flash-preview"


#Load once at module level (Shared across requests)
#This block of code run when main.py will launch server.
print("Laoding embaddig model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

print("Configurating Gemini...")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini = genai.GenerativeModel(GEMINI_MODEL)


def retrieve(question :str, top_k :int = TOP_K) -> list[dict]:
    """Embed the question and get the top_k most relavent chunks..."""
    question_embedding =  embedder.encode(question).tolist()

    result = collection.query(
        query_embeddings = [question_embedding],
        n_results= top_k,
        include=["documents","metadatas","distances"],

    )

    chunks = []
    for doc, meta, dist in zip(
        result["documents"][0], #Changed from 'results' to 'result'
        result["metadatas"][0],
        result["distances"][0],
    ):
        chunks.append({
            "text" : doc,
            "source" : meta.get("source","unknown"),
            "chunk_index" : meta.get("chunk_inde", -1),
            "distance": round(dist,4),
        })

    return chunks

def build_prompt(question :str, chunks: list[dict]) -> str:
    """Build the prompt the retrival context"""
    context_part =[]
    for i, chunk in enumerate(chunks,1):
        context_part.append(f"[{i}] (Source: {chunk['source']})\n{chunk['text']}")

    context = "\n\n".join(context_part)

    prompt =f""" You are a helpful assistant. Answer the user's question using only context provided below.
    If the answer not found in the context, say "I couldn't find relevent information in the provided documents."
    Donot make up information.  

    Context: {context}
    Question:{question}
    Answer :"""
    return prompt

def ask(question:str) -> dict:
    """Full RAG pipeline: retrive -> build prompt -> call Gemini -> return answer"""
    chunks= retrieve(question)

    if not chunks:
        return {
            "answer":"No relevent document found i  n database",
            "source": [],
        }
    prompt = build_prompt(question,chunks)
    response = gemini.generate_content(prompt)

    #Duplicate sources
    sources = list({chunk["source"] for chunk in chunks})
    #sources = list({chunk["source"] for chunk in chunks}) #for reff, from cloude
    print(f"\n-------{response.text.strip()}\n--------")
    return {
        "answer" : response.text.strip(),
        "sources": sources,
    } 