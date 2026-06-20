# src/processing/translator.py
import cyrtranslit
from langdetect import detect
import re

class ScriptTranslator:
    def __init__(self):
        self.cyrillic_pattern = re.compile(r'[а-яА-ЯёЁ]')
        
    def detect_script(self, text: str) -> str:
        """Detect if text contains Cyrillic"""
        if self.cyrillic_pattern.search(text):
            return "cyrillic"
        return "latinic"
    
    def cyrillic_to_latinic(self, text: str, lang: str = "sr") -> str:
        """
        Convert Cyrillic to Latinic
        Supports: 'sr' (Serbian), 'mk' (Macedonian), 'bg' (Bulgarian)
        """
        if self.detect_script(text) == "latinic":
            return text
            
        try:
            if lang == "sr":
                return cyrtranslit.to_latin(text, "sr")
            elif lang == "mk":
                return cyrtranslit.to_latin(text, "mk")
            else:
                # Auto-detect
                detected_lang = detect(text)
                if detected_lang in ['sr', 'mk', 'bg']:
                    return cyrtranslit.to_latin(text, detected_lang)
                else:
                    return text
        except Exception as e:
            # Fallback to Serbian
            return cyrtranslit.to_latin(text, "sr")
    
    def latinic_to_cyrillic(self, text: str, lang: str = "sr") -> str:
        """Convert Latinic to Cyrillic (if needed)"""
        try:
            if lang == "sr":
                return cyrtranslit.to_cyrillic(text, "sr")
            else:
                return cyrtranslit.to_cyrillic(text, "mk")
        except Exception:
            return text
    
    def normalize_query(self, text: str) -> str:
        """Normalize query by converting to Latinic"""
        return self.cyrillic_to_latinic(text)