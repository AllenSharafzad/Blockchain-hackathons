import os
import sys
import uuid
import re
from pathlib import Path
from openai import OpenAI
from pinecone import Pinecone
from PyPDF2 import PdfReader
from app.config import (
    OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV,
    PINECONE_INDEX_NAME, PINECONE_HOST
)

client = OpenAI(api_key=OPENAI_API_KEY)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PDF_PATH = DATA_DIR / "8a-code-of-practice-for-research-degrees.pdf"

def embed_text(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"  # Matches index dimension (1536)
    )
    return [r.embedding for r in response.data]

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def chunk_text(text, chunk_size=150, overlap=50):
    """
    Enhanced chunking with larger chunks and more overlap.
    Also uses sentence-aware chunking to avoid cutting sentences.
    """
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Split long sentences if needed
        words = sentence.split()
        
        if len(words) + current_length <= chunk_size:
            # Add the entire sentence if it fits
            current_chunk.extend(words)
            current_length += len(words)
        else:
            # If current chunk has content, add it to chunks
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Keep overlap words for context continuity
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:]
                    current_length = len(current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
            
            # Handle sentences that exceed chunk_size
            if len(words) > chunk_size:
                # Process the long sentence in segments
                for i in range(0, len(words), chunk_size - overlap):
                    segment = words[i:i + chunk_size]
                    chunks.append(" ".join(segment))
                    
                    # Keep last overlap words
                    if overlap > 0 and i + chunk_size < len(words):
                        current_chunk = segment[-overlap:]
                        current_length = len(current_chunk)
            else:
                # Start a new chunk with this sentence
                current_chunk = words
                current_length = len(words)
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def upload_to_pinecone(chunks, source_doc="8A Code of Practice", batch_size=50):
    print(f"Uploading {len(chunks)} chunks to Pinecone in batches of {batch_size}...")
    
    # Clear the index
    try:
        index.delete(delete_all=True)
        print("✅ Cleared existing Pinecone index.")
    except Exception as e:
        print(f"⚠️ Warning: Failed to clear index: {str(e)}. Proceeding with upsert.")

    embeddings = embed_text(chunks)
    to_upsert = [
        (
            f"chunk_{i}_{uuid.uuid4()}",
            embed,
            {
                "text": chunk,
                "source": source_doc,
                "notes": f"Chunk {i+1}"
            }
        )
        for i, (chunk, embed) in enumerate(zip(chunks, embeddings))
    ]

    # Upsert in batches
    for i in range(0, len(to_upsert), batch_size):
        batch = to_upsert[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"✅ Uploaded batch {i//batch_size + 1} ({len(batch)} chunks).")
        except Exception as e:
            print(f"❌ Error uploading batch {i//batch_size + 1}: {str(e)}")
    
    print("✅ Upload complete.")

def process_pdf():
    if not PDF_PATH.exists():
        print(f"❌ PDF not found at: {PDF_PATH}")
        return
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text)
    upload_to_pinecone(chunks)

if __name__ == "__main__":
    process_pdf()