"""
BM25 Retriever

Performs keyword-based retrieval using BM25.
"""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from config import CHROMA_DB_PATH
from utils.logger import logger


class BM25Retriever:

    def __init__(self):

        logger.info("Initializing BM25 Retriever...")

        self.index_path = Path(CHROMA_DB_PATH) / "bm25_index.pkl"

        self.documents_path = Path(CHROMA_DB_PATH) / "bm25_documents.pkl"

        # Create directory if it doesn't exist
        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.bm25 = None

        self.documents = []

        self.load_index()

    # -------------------------------------------------------

    def tokenize(
        self,
        text: str
    ):

        text = text.lower()

        return re.findall(
            r"\b\w+\b",
            text
        )

    # -------------------------------------------------------

    def build_index(
        self,
        chunks
    ):

        logger.info(
            "Building BM25 Index..."
        )

        corpus = []

        self.documents = []

        for chunk in chunks:

            corpus.append(
                self.tokenize(chunk.text)
            )

            self.documents.append(chunk)

        self.bm25 = BM25Okapi(corpus)

        with open(self.index_path, "wb") as f:

            pickle.dump(
                self.bm25,
                f
            )

        with open(self.documents_path, "wb") as f:

            pickle.dump(
                self.documents,
                f
            )

        logger.info(
            f"BM25 Index Saved ({len(self.documents)} chunks)."
        )

    # -------------------------------------------------------

    def load_index(self):

        if (

            self.index_path.exists()

            and

            self.documents_path.exists()

        ):

            logger.info(
                "Loading BM25 Index..."
            )

            with open(self.index_path, "rb") as f:

                self.bm25 = pickle.load(f)

            with open(self.documents_path, "rb") as f:

                self.documents = pickle.load(f)

            logger.info(
                f"Loaded BM25 index with {len(self.documents)} chunks."
            )

        else:

            logger.info(
                "BM25 index not found."
            )

    # -------------------------------------------------------

    def index_exists(self):

        return (

            self.index_path.exists()

            and

            self.documents_path.exists()

        )

    # -------------------------------------------------------

    def retrieve(
        self,
        question,
        top_k=5
    ):

        if self.bm25 is None:

            logger.warning(
                "BM25 index not loaded."
            )

            return []

        query_tokens = self.tokenize(question)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked = sorted(

            enumerate(scores),

            key=lambda x: x[1],

            reverse=True

        )

        results = []

        for idx, score in ranked[:top_k]:

            chunk = self.documents[idx]

            chunk.similarity_score = float(score)

            results.append(chunk)

        logger.info(
            f"BM25 retrieved {len(results)} chunks."
        )

        return results