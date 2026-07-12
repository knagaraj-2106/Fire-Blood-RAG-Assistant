"""
Production ChromaDB Manager
"""

import chromadb

from chromadb.config import Settings

from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
)

from utils.logger import logger


class ChromaManager:

    def __init__(self):
        print("=" * 60)
        print("Creating Chroma Client")
        print(CHROMA_DB_PATH)
        print("=" * 60)

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={
                "hnsw:space": "cosine"
            }
        )

    def collection_exists(self):

        count = self.collection.count()

        logger.info(
            f"Collection contains {count} vectors."
        )

        return count > 0

    def insert_chunks(self, chunks):

        logger.info("Uploading vectors...")

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in chunks:

            ids.append(chunk.chunk_id)

            documents.append(chunk.text)

            embeddings.append(chunk.embedding)

            metadatas.append({

                "page": chunk.page_number,

                "source": chunk.source,

                "chunk_index": chunk.chunk_index,

                "tokens": chunk.token_count

            })

        self.collection.add(

            ids=ids,

            documents=documents,

            embeddings=embeddings,

            metadatas=metadatas

        )

        logger.info(
            f"{len(ids)} vectors inserted."
        )

    def get_collection_count(self):

        return self.collection.count()