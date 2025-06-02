import os
from dotenv import load_dotenv

# Load from .env file
load_dotenv()

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_HOST = os.getenv("PINECONE_HOST")

# Retrieval Settings
TOP_K = int(os.getenv("TOP_K", 15))  # Back to 15 from 20
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.5))  # Back to 0.5 from 0.45
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "True").lower() == "true"

# Evaluation
QA_JSON_PATH = os.getenv("QA_JSON_PATH", "results/ragas_policy_only_results.json")

# Optional Neo4j credentials (commented out for now)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Debug/logging options
DEBUG_MODE = bool(os.getenv("DEBUG_MODE", False))