"""
Token Aware Chunking Engine
"""

import hashlib
import tiktoken
import os
import sys
PROJECT_ROOT = r"C:\Users\knagr\Downloads\Python Code FLM Sessions\Generative AI Projects\GEN_AI_PDF_CHATBOT"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from models.chunk import Chunk
from utils.logger import logger

from config import(CHUNK_SIZE,CHUNK_OVERLAP,TOKEN_ENCODING)

class Chunker:
    def __init__(self):
        self.encoding = tiktoken.get_encoding(TOKEN_ENCODING)
    def count_tokens(self,text: str)-> int:
        return len(self.encoding.encode(text))
    def generate_chunk_id(self,source,page_number,chunk_index,text):
        unique_string = (f"{source}_{page_number}_{chunk_index}_{text}")
        return hashlib.md5(unique_string.encode()).hexdigest()
    def chunk_documents(self, documents):

        logger.info("Chunking Started...")

        chunks = []

        for document in documents:

            tokens = self.encoding.encode(document.text)
            
            logger.info(f"Page {document.page_number} has {len(tokens)} tokens")

            start = 0

            chunk_index = 1

            while start < len(tokens):

                end = start + CHUNK_SIZE

                chunk_tokens = tokens[start:end]

                chunk_text = self.encoding.decode(chunk_tokens)

                chunk = Chunk(

    chunk_id=self.generate_chunk_id(

        document.source,
        document.page_number,
        chunk_index,
        chunk_text

    ),

    page_number=document.page_number,

    text=chunk_text,

    source=document.source,

    token_count=len(chunk_tokens),

    chunk_index=chunk_index,

    metadata={
        "page": document.page_number,
        "source": document.source
    }

)

                chunks.append(chunk)

                start += CHUNK_SIZE - CHUNK_OVERLAP

                chunk_index += 1

        logger.info(
            f"Generated {len(chunks)} chunks."
        )

        return chunks    
        