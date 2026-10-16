import chromadb
from pathlib import Path
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

BASE_DIR = Path(__file__).resolve().parent

# Load profile
with open(BASE_DIR / "profile.txt", "r", encoding="utf-8") as f:
    text = f.read()

#Split into chunks, each paragraph break (blank line) is a split point
#Strip removes extra space, and the if-check skips empty chunk

chunk = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()] 

print(f"Loaded {len(chunk)} chunks from profile.txt")


client = chromadb.PersistentClient(path="./chroma_db")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",  # Ollama's default local address
)

# Create (or reuse) a collection - like a table in a database
collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,  # Tells ChromaDB how to convert text to vectors
)

# Add chunks to the collection - ChromaDB automatically generates embeddings
collection.add(
    ids=[f"chunk{i}" for i in range(len(chunk))],  # Unique ID for each chunk
    documents=chunk,  # The actual text content
    metadatas=[{"source": "profile", "chunk_index": i} for i in range(len(chunk))],
)

print(f"Added {len(chunk)} chunks to the 'personal_profile' collection.")
print("Knowledge base built successfully!")