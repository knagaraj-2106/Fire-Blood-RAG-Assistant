"""
Embedding Generator

Generates embeddings using OpenAI's official SDK.
"""

import time
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    MAX_RETRIES,
)

from utils.logger import logger


class EmbeddingGenerator:

    def __init__(self):

        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _create_embedding(self, texts):

        retries = 0

        while retries < MAX_RETRIES:

            try:

                response = self.client.embeddings.create(

                    model=EMBEDDING_MODEL,

                    input=texts

                )

                return response.data

            except Exception as e:

                retries += 1

                logger.warning(

                    f"Embedding Retry {retries} : {e}"

                )

                time.sleep(2)

        raise Exception("Embedding generation failed.")

    def generate_embeddings(self, chunks):

        logger.info("Embedding Generation Started...")

        total_chunks = len(chunks)

        logger.info(f"Total Chunks : {total_chunks}")

        for batch_start in range(

            0,

            total_chunks,

            EMBEDDING_BATCH_SIZE

        ):

            batch = chunks[

                batch_start:

                batch_start + EMBEDDING_BATCH_SIZE

            ]

            texts = [

                chunk.text

                for chunk in batch

            ]

            embeddings = self._create_embedding(texts)

            for chunk, embedding in zip(batch, embeddings):

                chunk.embedding = embedding.embedding

        logger.info("Embedding Generation Completed.")

        return chunks