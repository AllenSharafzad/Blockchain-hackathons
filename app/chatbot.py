import os
import json
import logging
from datetime import datetime
from pathlib import Path
from app.retriever import retrieve_relevant_chunks
from openai import AsyncOpenAI
from app.config import OPENAI_API_KEY, DEBUG_MODE

# Set up logging
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    filename=str(log_file),
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chatbot")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def generate_chat_response(data):
    query = data.get("query", "")
    user_id = data.get("user_id", "default_user")
    history = data.get("history", [])
    
    logger.info(f"User {user_id} query: {query}")
    
    # Retrieve relevant chunks
    retrieval_result = retrieve_relevant_chunks(query)
    chunks = retrieval_result.get("chunks", [])
    retrieval_score = retrieval_result.get("retrieval_score", 0.0)
    source = retrieval_result.get("source", "")
    notes = retrieval_result.get("notes", "")
    chunk_sources = retrieval_result.get("chunk_sources", [])  # New field for detailed sources

    # Log retrieved chunks
    for i, (chunk, source_info) in enumerate(zip(chunks, chunk_sources if chunk_sources else [{"id": f"Chunk {i}", "source": source} for i in range(len(chunks))])):
        logger.debug(f"Retrieved chunk {i+1}: {source_info.get('id', f'Chunk {i+1}')} - {chunk[:100]}...")

    # Construct improved prompt for better context utilization and CONCISE answers
    numbered_chunks = []
    for i, (chunk, source_info) in enumerate(zip(chunks, chunk_sources if chunk_sources else [{"id": f"Chunk {i+1}", "source": source} for i in range(len(chunks))])):
        # Format: Context 1 (Section 2.3): Text content...
        context_header = f"Context {i+1}"
        if source_info.get("section"):
            context_header += f" (Section {source_info.get('section')})"
        elif source_info.get("page"):
            context_header += f" (Page {source_info.get('page')})"
        
        numbered_chunks.append(f"{context_header}: {chunk}")
    
    context = "\n\n".join(numbered_chunks) if chunks else "No relevant context found."
    
    prompt = f"""
Answer the following question about the Code of Practice for Research Degrees.

Question: {query}

Here is the relevant context from the document:
{context}

Instructions:
1. Provide a CONCISE answer (3-5 sentences if possible) based ONLY on the provided context.
2. Format the answer for readability using bullet points where appropriate.
3. If the context doesn't contain enough information, briefly state what's missing.
4. Reference specific contexts by their numbers (e.g., "According to Context 3...").
5. Focus on giving the most important information first.
"""

    # Generate response
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specifically trained to provide CONCISE answers about Bournemouth University's Code of Practice for Research Degrees. Prioritize brevity and clarity over comprehensiveness. Keep responses short and focused on the main points."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350  # Reduced slightly but not too much to preserve quality
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"Generated answer: {answer[:100]}...")
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        answer = f"Error generating response: {str(e)}"

    # Debug logging
    logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
    logger.info(f"Retrieval score: {retrieval_score}")
    
    # Save complete interaction log to file for traceability
    interaction_log = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "query": query,
        "chunks": chunks,
        "chunk_sources": chunk_sources if chunk_sources else [{"id": f"Chunk {i}", "source": source} for i in range(len(chunks))],
        "retrieval_score": retrieval_score,
        "answer": answer
    }
    
    interaction_log_file = log_dir / f"interactions_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(interaction_log_file, "a") as f:
        f.write(json.dumps(interaction_log) + "\n")

    # Return the response with important metadata
    return {
        "answer": answer,
        "chunks": chunks,
        "chunk_sources": chunk_sources if chunk_sources else [{"id": f"Chunk {i+1}", "source": source} for i in range(len(chunks))],
        "retrieval_score": retrieval_score,
        "source": source,
        "notes": notes
    }