"""
Production Indexing Pipeline

Responsible for:

PDF
    ↓
Documents
    ↓
Chunks
    ↓
Embeddings
    ↓
BM25 Index
    ↓
ChromaDB
"""

from config import PDF_PATH

from ingestion.pdf_loader import PDFLoader
from ingestion.chunker import Chunker

from embedding.embedding_generator import EmbeddingGenerator

from vectordb.chroma_manager import ChromaManager
from retrieval.bm25_retriever import BM25Retriever

from utils.logger import logger


class IndexingPipeline:

    def __init__(self):

        self.loader = PDFLoader(PDF_PATH)

        self.chunker = Chunker()

        self.embedding_generator = EmbeddingGenerator()

        self.vector_db = ChromaManager()

        self.bm25 = BM25Retriever()

    # ---------------------------------------------------------

    def build_vector_database(self):

        """
        Builds both:

        1. ChromaDB Vector Store

        2. BM25 Keyword Index
        """

        vector_exists = self.vector_db.collection_exists()

        bm25_exists = self.bm25.index_exists()
        
        if vector_exists and bm25_exists:
        
            logger.info(
                "Vector Database and BM25 Index already exist."
            )
        
            return False

        # -----------------------------------------------------
        # Load PDF
        # -----------------------------------------------------

        logger.info("Loading PDF...")

        documents = self.loader.load_pdf()

        logger.info(
            f"Loaded {len(documents)} pages."
        )

        # -----------------------------------------------------
        # Chunk Documents
        # -----------------------------------------------------

        logger.info("Chunking documents...")

        chunks = self.chunker.chunk_documents(
            documents
        )

        logger.info(
            f"Created {len(chunks)} chunks."
        )

        # -----------------------------------------------------
        # Generate Embeddings
        # -----------------------------------------------------

        logger.info(
            "Generating embeddings..."
        )

        embedded_chunks = (

            self.embedding_generator.generate_embeddings(
                chunks
            )

        )

        logger.info(
            "Embeddings generated successfully."
        )

        # -----------------------------------------------------
        # Build BM25
        # -----------------------------------------------------

        logger.info(
            "Building BM25 Index..."
        )

        self.bm25.build_index(
            embedded_chunks
        )

        logger.info(
            "BM25 Index Created Successfully."
        )

        # -----------------------------------------------------
        # Insert into Chroma
        # -----------------------------------------------------

        logger.info(
            "Uploading vectors to ChromaDB..."
        )

        self.vector_db.insert_chunks(
            embedded_chunks
        )

        logger.info(
            "Vector Database Created Successfully."
        )

        return True