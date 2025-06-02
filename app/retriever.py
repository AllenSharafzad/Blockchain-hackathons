import os
import re
import logging
from openai import OpenAI
from pinecone import Pinecone
from app.config import (
    OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV,
    PINECONE_INDEX_NAME, TOP_K, MIN_SCORE, PINECONE_HOST, DEBUG_MODE
)

# Set up logging
logger = logging.getLogger("retriever")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Regular expression to extract section numbers from text
SECTION_PATTERN = re.compile(r'Section (\d+\.\d+(?:\.\d+)?)')
PAGE_PATTERN = re.compile(r'Page (\d+)')

def embed_query(query):
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def generate_query_variations(query):
    """Generate variations of the query to improve recall"""
    try:
        # Use OpenAI to generate query reformulations
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that reformulates research queries to improve retrieval. Generate 2-3 alternative phrasings that preserve the original meaning but use different terminology. Format as a Python list."},
                {"role": "user", "content": f"Original query: {query}\nGenerate alternative phrasings:"}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        # Extract and clean the response
        content = response.choices[0].message.content.strip()
        
        # Simple parsing to extract list items
        import re
        variations = re.findall(r'"([^"]*)"', content) or re.findall(r"'([^']*)'", content)
        
        if not variations:
            # Try to extract lines that might be list items
            variations = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('[') and not line.strip().endswith(']')]
        
        # Add the original query and return unique variations
        all_queries = [query] + variations
        return list(set(all_queries))
    
    except Exception as e:
        logger.error(f"Error generating query variations: {str(e)}")
        return [query]  # Fallback to original query

def extract_section_info(text):
    """Extract section numbers or page numbers from text"""
    # Try to find section numbers
    section_match = SECTION_PATTERN.search(text)
    if section_match:
        return {"section": section_match.group(1)}
    
    # Try to find page numbers
    page_match = PAGE_PATTERN.search(text)
    if page_match:
        return {"page": page_match.group(1)}
    
    return {}

def retrieve_relevant_chunks(query, lower_min_score=0.45):
    """
    Enhanced retrieval with query expansion, dynamic scoring,
    and improved source attribution.
    """
    logger.info(f"Original query: {query}")
    logger.info(f"Using TOP_K={TOP_K}, MIN_SCORE={MIN_SCORE}")
    
    # Generate query variations to improve recall
    query_variations = generate_query_variations(query)
    logger.info(f"Query variations: {query_variations}")
    
    all_results = []
    
    # Get results for each query variation
    for q in query_variations:
        embedding = embed_query(q)
        
        results = index.query(
            vector=embedding,
            top_k=TOP_K,
            include_metadata=True
        )
        
        # Add results to the pool
        all_results.extend(results.matches)
    
    # Remove duplicates by ID
    seen_ids = set()
    unique_results = []
    for match in all_results:
        if match.id not in seen_ids:
            unique_results.append(match)
            seen_ids.add(match.id)
    
    # Sort by score
    unique_results.sort(key=lambda x: x.score, reverse=True)
    
    logger.info(f"🔍 Total unique matches returned: {len(unique_results)}")
    for i, m in enumerate(unique_results[:10]):
        snippet = m.metadata.get("text", "")[:80].replace("\n", " ")
        logger.info(f"→ Score: {m.score:.4f} | Preview: {snippet}")
    
    # First try with normal MIN_SCORE
    filtered = [match for match in unique_results if match.score >= MIN_SCORE]
    
    # If we don't have enough results, try with a lower threshold
    if len(filtered) < 3:
        logger.info(f"Not enough results with MIN_SCORE={MIN_SCORE}, trying with lower threshold {lower_min_score}")
        filtered = [match for match in unique_results if match.score >= lower_min_score]
    
    # If still no results, take the top 5 regardless of score
    if not filtered:
        logger.info("No results above threshold, taking top 5 results")
        filtered = unique_results[:5]
    
    # Get text and metadata with enhanced source information
    chunks = []
    chunk_scores = []
    chunk_sources = []
    sources = set()
    notes = set()
    
    for i, match in enumerate(filtered):
        text = match.metadata.get("text", "").strip()
        if text:
            # Extract chunk ID from the match ID
            chunk_id = match.id.split('_')[1] if '_' in match.id else str(i+1)
            
            # Extract metadata
            source = match.metadata.get("source", "unknown")
            note = match.metadata.get("notes", "")
            
            # Try to extract section number from text or notes
            section_info = extract_section_info(text) or extract_section_info(note)
            
            # Create enhanced source info
            source_info = {
                "id": f"Chunk {chunk_id}",
                "source": source,
            }
            if section_info.get("section"):
                source_info["section"] = section_info["section"]
            elif section_info.get("page"):
                source_info["page"] = section_info["page"]
                
            if note:
                source_info["note"] = note
            
            # Add to our collections
            chunks.append(text)
            chunk_scores.append(match.score)
            chunk_sources.append(source_info)
            
            if source:
                sources.add(source)
                
            if note:
                notes.add(note)
    
    # Ensure we don't have too many chunks
    max_chunks = 15
    if len(chunks) > max_chunks:
        # Keep top scoring chunks
        chunk_indices = sorted(range(len(chunks)), key=lambda i: chunk_scores[i], reverse=True)[:max_chunks]
        chunks = [chunks[i] for i in chunk_indices]
        chunk_sources = [chunk_sources[i] for i in chunk_indices]
        chunk_scores = [chunk_scores[i] for i in chunk_indices]
    
    retrieval_score = chunk_scores[0] if chunk_scores else 0.0
    
    return {
        "chunks": chunks,
        "chunk_sources": chunk_sources,  # Enhanced source information
        "retrieval_score": retrieval_score,
        "source": ", ".join(sources),
        "notes": "; ".join(notes)
    }