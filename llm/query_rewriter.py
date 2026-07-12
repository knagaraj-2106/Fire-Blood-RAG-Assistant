"""
Query Rewriter

Rewrites conversational follow-up questions into
standalone questions before semantic retrieval.
"""

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    CHAT_MODEL
)

from utils.logger import logger


class QueryRewriter:

    def __init__(self):

        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

    def rewrite(
        self,
        question: str,
        chat_history: list
    ) -> str:

        logger.info("Rewriting user question...")

        history = ""

        for message in chat_history:

            history += (
                f"{message['role'].capitalize()}: "
                f"{message['content']}\n"
            )

        system_prompt = """
You are an expert query rewriting assistant.

Your job is to convert the user's latest question into a
fully self-contained standalone question.

Rules:

1. Preserve the original meaning.
2. Resolve pronouns like:
   - he
   - she
   - him
   - her
   - they
   - it
3. Never answer the question.
4. Return ONLY the rewritten question.
5. If the question is already standalone,
   return it unchanged.
"""

        user_prompt = f"""
Conversation History:

{history}

Latest User Question:

{question}

Rewrite the latest question as a standalone question.
"""

        response = self.client.chat.completions.create(

            model=CHAT_MODEL,

            temperature=0,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": user_prompt
                }

            ]

        )

        rewritten_question = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        logger.info(
            f"Standalone Question: {rewritten_question}"
        )

        return rewritten_question