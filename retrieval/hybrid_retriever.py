"""
Production Hybrid Retriever

Combines:

1. Semantic Retrieval (ChromaDB)

2. BM25 Retrieval

using Weighted Reciprocal Rank Fusion (RRF).
"""

import time

from retrieval.semantic_retriever import SemanticRetriever
from retrieval.bm25_retriever import BM25Retriever

from utils.logger import logger


class HybridRetriever:

    # Higher weight -> more influence
    SEMANTIC_WEIGHT = 0.70
    BM25_WEIGHT = 0.30

    # Standard RRF constant
    RRF_K = 60

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing Hybrid Retriever")
        logger.info("=" * 60)

        self.semantic = SemanticRetriever()
        self.bm25 = BM25Retriever()

        logger.info("Hybrid Retriever Ready")

    # ---------------------------------------------------------

    def _add_rrf_scores(
        self,
        results,
        weight,
        score_table
    ):
        """
        Adds Reciprocal Rank Fusion scores.
        """

        for rank, chunk in enumerate(results):

            rrf_score = weight / (self.RRF_K + rank + 1)

            if chunk.chunk_id not in score_table:

                score_table[chunk.chunk_id] = {

                    "chunk": chunk,

                    "score": 0.0

                }

            score_table[chunk.chunk_id]["score"] += rrf_score

    # ---------------------------------------------------------

    def retrieve(
        self,
        question
    ):

        start = time.perf_counter()

        logger.info("Running Semantic Retrieval...")

        semantic_result = self.semantic.retrieve(question)

        semantic_chunks = semantic_result["chunks"]

        logger.info(
            f"Semantic Retrieved: {len(semantic_chunks)}"
        )

        logger.info("Running BM25 Retrieval...")

        bm25_chunks = self.bm25.retrieve(question)

        logger.info(
            f"BM25 Retrieved: {len(bm25_chunks)}"
        )

        # -----------------------------------------------------
        # Reciprocal Rank Fusion
        # -----------------------------------------------------

        score_table = {}

        self._add_rrf_scores(
            semantic_chunks,
            self.SEMANTIC_WEIGHT,
            score_table
        )

        self._add_rrf_scores(
            bm25_chunks,
            self.BM25_WEIGHT,
            score_table
        )

        ranked = sorted(

            score_table.values(),

            key=lambda x: x["score"],

            reverse=True

        )

        final_chunks = []

        for item in ranked:

            chunk = item["chunk"]

            # Store fused score for display
            chunk.similarity_score = round(
                item["score"],
                4
            )

            final_chunks.append(chunk)

        retrieval_time = round(

            time.perf_counter() - start,

            4

        )

        logger.info(
            f"Hybrid Retrieval completed in "
            f"{retrieval_time:.4f} sec"
        )

        return {

            "chunks": final_chunks,

            "retrieval_time": retrieval_time,

            "retrieved_chunks": len(final_chunks),

            "top_similarity": (
                final_chunks[0].similarity_score
                if final_chunks
                else 0.0
            )

        }