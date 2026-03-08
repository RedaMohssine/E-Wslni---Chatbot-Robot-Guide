"""
Embed document chunks using Google Gemini and store in ChromaDB.
"""

import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chunker import load_all_data

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "emines_rag"
EMBEDDING_MODEL = "models/gemini-embedding-001"
BATCH_SIZE = 80
PAUSE_SECONDS = 65


def setup_api():
    """Load API key and configure Gemini."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
        print("ERROR: Please set GOOGLE_API_KEY in the .env file.")
        exit(1)
    genai.configure(api_key=api_key)
    print("API key loaded.")
    return api_key


def embed_texts(texts):
    """Embed a batch of document texts."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
        task_type="retrieval_document",
    )
    return result['embedding']


def embed_query(query_text):
    """Embed a query (uses retrieval_query task type)."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query_text,
        task_type="retrieval_query",
    )
    return result['embedding']


def create_vector_store(chunks):
    """Embed all chunks and store in ChromaDB."""
    print(f"\nEmbedding {len(chunks)} chunks into ChromaDB...")
    
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "EMINES/UM6P RAG knowledge base"}
    )
    
    print(f"Collection '{COLLECTION_NAME}' created at: {CHROMA_DB_DIR}/")
    
    start_time = time.time()
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        
        # Extract texts from this batch
        texts = [chunk["text"] for chunk in batch]
        
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        
        metadatas = []
        for chunk in batch:
            meta = {}
            for key, value in chunk["metadata"].items():
                meta[key] = str(value)
            metadatas.append(meta)
        
        try:
            embeddings = embed_texts(texts)
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            
            print(f"  Batch {batch_num}/{total_batches} - {len(batch)} chunks")
            
        except Exception as e:
            print(f"  Batch {batch_num} rate limited, waiting 65s...")
            time.sleep(65)
            # Retry
            embeddings = embed_texts(texts)
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            print(f"  Batch {batch_num}/{total_batches} - retry succeeded")
        if i + BATCH_SIZE < len(chunks):
            time.sleep(PAUSE_SECONDS)
    
    elapsed = time.time() - start_time
    print(f"Done. {len(chunks)} chunks embedded in {elapsed:.1f}s")
    
    return client, collection


def test_retrieval(collection):
    """Test semantic search with sample questions."""
    print("\nTesting retrieval...")
    
    test_questions = [
        "Quels sont les frais de scolarité à l'EMINES ?",
        "How can I apply to the engineering program?",
        "Où se trouve le campus ?",
        "Quelles sont les options en 3ème année ?",
        "C'est quoi le club E-Tech ?",
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        print("-" * 50)
        
        query_embedding = embed_query(question)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
        )
        
        for i in range(len(results["documents"][0])):
            doc_text = results["documents"][0][i]
            source = results["metadatas"][0][i].get("source", "?")
            distance = results["distances"][0][i]
            
            preview = doc_text[:150].replace("\n", " ")
            print(f"  [{i+1}] (dist: {distance:.4f}) {source[:60]}")
            print(f"      {preview}...")
        
        time.sleep(1)


if __name__ == "__main__":
    setup_api()
    chunks = load_all_data()
    client, collection = create_vector_store(chunks)
    test_retrieval(collection)
    print(f"\nVector database ready at: {CHROMA_DB_DIR}/")
