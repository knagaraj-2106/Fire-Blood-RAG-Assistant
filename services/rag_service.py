"""
Production RAG Service

Coordinates:

1. Query Rewriting
2. Hybrid Retrieval
3. Prompt Construction
4. LLM Generation
"""

import time

from retrieval.hybrid_retriever import HybridRetriever
from llm.query_rewriter import QueryRewriter
from llm.prompt_builder import PromptBuilder
from llm.llm_manager import LLMManager

from utils.logger import logger


class RAGService:

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing RAG Service")
        logger.info("=" * 60)

        self.retriever = HybridRetriever()

        self.query_rewriter = QueryRewriter()

        self.prompt_builder = PromptBuilder()

        self.llm = LLMManager()

        logger.info("RAG Service initialized successfully.")

    # ---------------------------------------------------------

    def answer(self, question):

        return self.ask(question, [])

    # ---------------------------------------------------------

    def ask(
        self,
        question,
        chat_history
    ):

        total_start = time.perf_counter()

        logger.info(f"Original Question: {question}")

        # -------------------------------------------------
        # Query Rewriting
        # -------------------------------------------------

        rewritten_question = self.query_rewriter.rewrite(
            question,
            chat_history
        )

        logger.info(
            f"Rewritten Question: {rewritten_question}"
        )

        # -------------------------------------------------
        # Hybrid Retrieval
        # -------------------------------------------------

        retrieval_result = self.retriever.retrieve(
            rewritten_question
        )

        chunks = retrieval_result["chunks"]

        retrieval_time = retrieval_result["retrieval_time"]

        retrieved_chunks = retrieval_result["retrieved_chunks"]

        top_similarity = retrieval_result["top_similarity"]

        # -------------------------------------------------
        # No Context Found
        # -------------------------------------------------

        if not chunks:

            total_time = round(
                time.perf_counter() - total_start,
                4
            )

            logger.warning("No relevant chunks retrieved.")

            return {

                "status": "no_context",

                "original_question": question,

                "rewritten_question": rewritten_question,

                "answer":
                "I couldn't find the answer in the indexed document.",

                "sources": [],

                "retrieval_time": retrieval_time,

                "generation_time": 0.0,

                "total_time": total_time,

                "retrieved_chunks": 0,

                "top_similarity": 0.0

            }

        # -------------------------------------------------
        # Prompt Building
        # -------------------------------------------------

        system_prompt, user_prompt = (

            self.prompt_builder.build_prompt(

                chat_history,

                question,

                rewritten_question,

                chunks

            )

        )

        # -------------------------------------------------
        # LLM Generation
        # -------------------------------------------------

        generation_start = time.perf_counter()

        answer = self.llm.generate_answer(

            system_prompt,

            user_prompt

        )

        generation_time = round(

            time.perf_counter() - generation_start,

            4

        )

        total_time = round(

            time.perf_counter() - total_start,

            4

        )

        logger.info(
            f"Retrieval Time : {retrieval_time:.4f} sec"
        )

        logger.info(
            f"Generation Time : {generation_time:.4f} sec"
        )

        logger.info(
            f"Total Time : {total_time:.4f} sec"
        )

        return {

            "status": "success",

            "original_question": question,

            "rewritten_question": rewritten_question,

            "answer": answer,

            "sources": chunks,

            "retrieval_time": retrieval_time,

            "generation_time": generation_time,

            "total_time": total_time,

            "retrieved_chunks": retrieved_chunks,

            "top_similarity": top_similarity

        }