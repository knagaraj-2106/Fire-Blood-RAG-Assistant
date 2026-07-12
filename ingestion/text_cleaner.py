"""
Text Cleaning Module
"""
import re
class TextCleaner:

    @staticmethod
    def clean_text(text:str) -> str:
        """
        Clean extracted PDF text.
        """
        if not text:
            return ""
        #Remove extra spaces
        text = re.sub(r"[ \t]+", " ", text)

        #Remove multiple blank lines
        text = re.sub(r"\n{2,}","\n", text)

        #Remove page numbers standing alone
        text = re.sub(r"\n\d+\n","\n", text)

        #Remove leading/trailing spaces
        text = text.strip()

        return text