"""
Production Retrieval Pipeline
"""

from retrieval.retriever import Retriever

from utils.logger import logger


class RetrievalPipeline:

    def __init__(self):

        self.retriever = Retriever()

    def retrieve(self, question):

        logger.info(
            "Starting Retrieval Pipeline..."
        )

        chunks = self.retriever.retrieve(
            question
        )

        logger.info(
            f"{len(chunks)} chunks returned."
        )

        return chunks