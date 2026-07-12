"""
Prompt Builder

Builds the prompt that will be sent to GPT.
"""

from typing import List

from models.retrieved_chunk import RetrievedChunk


class PromptBuilder:

    SYSTEM_PROMPT = """
You are a helpful AI assistant specialized in answering questions ONLY from the provided document context.

Rules:

1. Use ONLY the supplied context.
2. Never use outside knowledge.
3. If the answer is not present in the context, respond exactly:
   "I couldn't find the answer in the provided document."
4. If multiple pages support the answer, mention all relevant page numbers.
5. Keep answers factual and concise.
6. Never hallucinate or assume missing information.
"""

    @classmethod
    def build_prompt(
        cls,
        chat_history,
        original_question,
        rewritten_question,
        chunks: List[RetrievedChunk]
    ):

        # ------------------------------------------
        # Conversation History
        # ------------------------------------------

        history = ""

        for message in chat_history:

            history += (
                f"{message['role'].capitalize()}: "
                f"{message['content']}\n"
            )

        # ------------------------------------------
        # Retrieved Context
        # ------------------------------------------

        context = ""

        for chunk in chunks:

            context += (
                f"\n\n"
                f"Page Number: {chunk.page_number}\n"
                f"{chunk.text}"
            )

        # ------------------------------------------
        # User Prompt
        # ------------------------------------------

        user_prompt = f"""
Conversation History
====================

{history}

------------------------------------------------

Original User Question

{original_question}

------------------------------------------------

Interpreted Standalone Question

{rewritten_question}

------------------------------------------------

Retrieved Context

{context}

------------------------------------------------

Instructions

Answer the ORIGINAL USER QUESTION using ONLY the retrieved context.
Mention page numbers whenever possible.
"""

        return cls.SYSTEM_PROMPT, user_prompt