"""
Global configuration file for the projects.

Author : Nagaraj K

Project : Production RAG Chatbot
"""

from pathlib import Path
import os

#Base Directory
BASE_DIR = Path(__file__).resolve().parent

#Data
DATA_DIR = BASE_DIR/'data'
PDF_NAME = 'Fire_and_Blood.pdf'
PDF_PATH = DATA_DIR / PDF_NAME

#ChromaDB
CHROMA_DB_PATH = BASE_DIR/'chroma_db'
CHROMA_COLLECTION = "fire_and_blood"
CHROMA_DISTANCE = "cosine"

#Embedding Model
EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_BATCH_SIZE = 50
EMBEDDING_DIMENSION = 1536
MAX_RETRIES = 3

#OpenAI Chat Model
CHAT_MODEL = 'gpt-4o-mini'

#Chunk configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOKEN_ENCODING = "cl100k_base"

#Retrieval Configuration
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.35

#Logging
LOG_DIR = BASE_DIR/'logs'
LOG_FILE = LOG_DIR/'rag_application.log'

#OpenAI Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# LLM Configuration
LLM_TEMPERATURE = 0
MAX_OUTPUT_TOKENS = 800