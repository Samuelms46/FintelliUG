import re
import string
from langdetect import detect, LangDetectException
from config import Config

class DataCleaner:
    def __init__(self):
        self.stopwords = set()  # Could load custom stopwords for Uganda context
    
    def clean_text(self, text):
        """Clean social media text by removing noise"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove mentions and hashtags but keep the text
        text = re.sub(r'@(\w+)', r'\1', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_language(self, text):
        """Detect language with fallback to English"""
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return "en"  # Default to English
    
    def is_relevant(self, text, min_length=10):
        """Check if text is relevant for fintech analysis"""
        if not text or len(text) < min_length:
            return False
        
        # Check for fintech-related keywords (case insensitive)
        fintech_keywords = [
            'money', 'bank', 'loan', 'save', 'invest', 'payment',
            'mobile', 'digital', 'fintech', 'cash', 'transfer', 
            'mtn', 'airtel', 'uganda', 'ugx'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in fintech_keywords if keyword in text_lower)
        
        # Consider text relevant if it contains at least 2 fintech keywords
        return keyword_count >= 2
    
    def calculate_relevance_score(self, text):
        """Calculate a relevance score based on keyword presence"""
        if not text:
            return 0.0
        
        fintech_keywords = [
            'money', 'bank', 'loan', 'save', 'invest', 'payment',
            'mobile', 'digital', 'fintech', 'cash', 'transfer', 
            'mtn', 'airtel', 'uganda', 'ugx'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in fintech_keywords if keyword in text_lower)
        
        # Normalize score between 0 and 1
        max_possible = len(fintech_keywords)
        return min(keyword_count / 10, 1.0)  # Cap at 1.0 even if many keywords