"""
Production Semantic Retriever

Performs semantic similarity search
using OpenAI Embeddings + ChromaDB.
"""

import time

from openai import OpenAI

import chromadb
from chromadb.config import Settings

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD
)

from models.retrieved_chunk import RetrievedChunk
from utils.logger import logger


class SemanticRetriever:

    def __init__(self):

        logger.info("Initializing Semantic Retriever...")

        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        chroma = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        try:

            self.collection = chroma.get_collection(
                CHROMA_COLLECTION
            )

            logger.info(
                f"Connected to Chroma Collection: {CHROMA_COLLECTION}"
            )

        except Exception as e:

            logger.exception(
                "Unable to open Chroma collection."
            )

            raise RuntimeError(
                f"Collection '{CHROMA_COLLECTION}' not found. "
                "Please rebuild the vector database."
            ) from e

    # ---------------------------------------------------------

    def embed_query(
        self,
        question: str
    ):

        logger.info("Generating query embedding...")

        response = self.client.embeddings.create(

            model=EMBEDDING_MODEL,

            input=question

        )

        return response.data[0].embedding

    # ---------------------------------------------------------

    def retrieve(
        self,
        question
    ):

        start = time.perf_counter()

        logger.info(
            "Performing semantic search..."
        )

        query_embedding = self.embed_query(
            question
        )

        results = self.collection.query(

            query_embeddings=[query_embedding],

            n_results=TOP_K_RESULTS,

            include=[
                "documents",
                "metadatas",
                "distances"
            ]

        )

        chunks = []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, meta, distance in zip(

            ids,

            docs,

            metas,

            distances

        ):

            similarity = 1 - float(distance)

            if similarity < SIMILARITY_THRESHOLD:

                continue

            chunk = RetrievedChunk(

                chunk_id=chunk_id,

                text=text,

                page_number=meta["page"],

                source=meta["source"],

                similarity_score=round(similarity, 4),

                chunk_index=meta["chunk_index"]

            )

            chunks.append(chunk)

        retrieval_time = round(

            time.perf_counter() - start,

            4

        )

        logger.info(
            f"Semantic Search completed in "
            f"{retrieval_time:.4f} sec"
        )

        logger.info(
            f"Retrieved {len(chunks)} semantic chunks."
        )

        top_similarity = (

            chunks[0].similarity_score

            if chunks

            else 0.0

        )

        return {

            "chunks": chunks,

            "retrieval_time": retrieval_time,

            "retrieved_chunks": len(chunks),

            "top_similarity": top_similarity

        }