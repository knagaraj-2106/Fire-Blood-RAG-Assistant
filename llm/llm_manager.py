"""
LLM Manager

Handles all communication with GPT.
"""

from openai import OpenAI

from config import (

    OPENAI_API_KEY,

    CHAT_MODEL,

    LLM_TEMPERATURE,

    MAX_OUTPUT_TOKENS

)

from utils.logger import logger


class LLMManager:

    def __init__(self):

        self.client = OpenAI(

            api_key=OPENAI_API_KEY

        )

    def generate_answer(

        self,

        system_prompt,

        user_prompt

    ):

        logger.info(

            "Sending Prompt to GPT..."

        )

        response = self.client.chat.completions.create(

            model=CHAT_MODEL,

            temperature=LLM_TEMPERATURE,

            max_completion_tokens=MAX_OUTPUT_TOKENS,

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

        answer = (

            response

            .choices[0]

            .message

            .content

        )

        logger.info(

            "LLM Response Generated."

        )

        return answer