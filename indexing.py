from dotenv import load_dotenv
import os
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

load_dotenv()
#Config
PDF_DIR = "./pdfs"
CHOROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
OVERLAP_SIZE = 50

# 1.Extracting text data from pdf
def extract_text_from_pdf(pdf_path: str) ->str:
    """Extract all texts from pdf file"""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

# 2.Chunking Extracted data
def chunk_text(text : str, chunk_size : int = CHUNK_SIZE,overlap_size: int = OVERLAP_SIZE) -> list[str]:
    """Split chunks into overlapping chunks"""
    chunks =[]
    start = 0
    while(start < len(text)):
        end = start + chunk_size
        chunk = text[start:end].strip() #Chunks created here.
        if chunk :
            chunks.append(chunk)
        start += chunk_size - overlap_size
    return chunks    

def ingest():
    #1 load embadding model
    print(f"loading embadding model {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    #2.connect to ChromaDB
    client = chromadb.PersistentClient(path =CHOROMA_DIR)

    #Delete exisitng connection
    exisiting = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in exisiting:
        print(f"deleteing existing connection {COLLECTION_NAME}...")
        client.delete_collection(COLLECTION_NAME)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    #3. Process each pdf
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No pdf file found in {PDF_DIR}.")
        return

    all_chunks = []
    all_ids =[]
    all_metadata = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR,pdf_file)
        print(f"processing {pdf_file}.")    

        text = extract_text_from_pdf(pdf_path) # calling fn for pdf extraction 
        if not text.strip() :
            print(f"no text extracted from {pdf_file}...")
            continue
        chunks = chunk_text(text) #calling fn for chunk creation
        print(f"{len(chunks)} created.")

        for i,chunk in enumerate(chunks):
            chunk_id = f"{pdf_file}_chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({"source": pdf_file,"chunk_index":i})

    if not all_chunks:
        print("no chunks to ingest, exiting...")
        return

    #4. Embadding 
    print (f"generating embadding for {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist() # here embedding happenes 

    #5 Store in ChromaDB
    BATCH_SIZE =100
    for i in range(0, len(all_chunks), BATCH_SIZE):
        collection.add(
            documents= all_chunks[i:i+BATCH_SIZE],
            embeddings = embeddings[i:i+BATCH_SIZE],
            ids = all_ids[i:i+BATCH_SIZE],
            metadatas = all_metadata[i:i+BATCH_SIZE]
        )
    print(f"\nIngestion complete. {len(all_chunks)} chunks stored in ChromaDB")
    print(f" Collection: {COLLECTION_NAME} |  Path : '{CHOROMA_DIR}'")

if __name__ == "__main__":
    ingest()