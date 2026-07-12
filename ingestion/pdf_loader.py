"""
PDF Loader

Reads PDF page-by-page using PyMuPDF.
"""

import fitz

from pathlib import Path

from models.document import Document
from ingestion.text_cleaner import TextCleaner
from utils.logger import logger


class PDFLoader:

    def __init__(self, pdf_path: Path):

        self.pdf_path = pdf_path

    def load_pdf(self):

        documents = []

        logger.info("Opening PDF")

        pdf = fitz.open(self.pdf_path)

        logger.info(f"Total Pages : {pdf.page_count}")

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text()

            cleaned_text = TextCleaner.clean_text(text)

            if cleaned_text:

                document = Document(

                    page_number=page_number,

                    text=cleaned_text,

                    source=self.pdf_path.name

                )

                documents.append(document)

        pdf.close()

        logger.info(f"Loaded {len(documents)} pages successfully")

        return documents