import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.cleaner import DataCleaner
from data_processing.topic_extractor import TopicExtractor
from database.db_manager import DatabaseManager
from database.vector_db import ChromaDBManager
from database.models import SocialMediaPost
from config import Config

class TestDataProcessing:
    def setup_method(self):
        self.cleaner = DataCleaner()
        self.topic_extractor = TopicExtractor()
        self.db_manager = DatabaseManager()
        self.vector_db = ChromaDBManager()
        
    def test_text_cleaning(self):
        """Test that text cleaning works correctly"""
        # Test with realistic social media content
        test_cases = [
            {
                "input": "Check this https://example.com and http://test.com #fintech",
                "expected_clean": "Check this and fintech",
                "description": "Should remove URLs and hashtags but keep text"
            },
            {
                "input": "MTN Mobile Money is great! @user123 😊",
                "expected_clean": "MTN Mobile Money is great user123",
                "description": "Should remove mentions and emojis but keep text"
            },
            {
                "input": "I   have   multiple   spaces   and   special/chars!",
                "expected_clean": "I have multiple spaces and special chars",
                "description": "Should normalize whitespace and remove special chars"
            }
        ]
        
        for case in test_cases:
            cleaned = self.cleaner.clean_text(case["input"])
            assert cleaned == case["expected_clean"], case["description"]
    
    def test_relevance_detection_uganda_fintech(self):
        """Test relevance detection specifically for Uganda fintech content"""
        # Relevant Uganda fintech content
        relevant_cases = [
            "MTN Mobile Money charges 1000 UGX for transactions",
            "Airtel Money has better coverage in rural Uganda",
            "Bank of Uganda announces new fintech regulations",
            "Send money to Uganda with WorldRemit",
            "Mobile lending apps like Okash are popular in Kampala"
        ]
        
        # Irrelevant content
        irrelevant_cases = [
            "Weather in Kampala is nice today",
            "Football match between Uganda and Kenya",
            "Restaurant review: best food in Entebbe",
            "I bought new clothes at Acacia Mall"
        ]
        
        for text in relevant_cases:
            assert self.cleaner.is_relevant(text), f"Should be relevant: {text}"
            
        for text in irrelevant_cases:
            assert not self.cleaner.is_relevant(text), f"Should be irrelevant: {text}"
    
    def test_topic_extraction_fintech_categories(self):
        """Test topic extraction for fintech categories"""
        test_cases = [
            {
                "text": "MTN Mobile Money fees are too high for sending money",
                "expected_topics": ["Mobile Money"],
                "description": "Should detect Mobile Money topic"
            },
            {
                "text": "Stanbic Bank offers new digital banking services with mobile app",
                "expected_topics": ["Digital Banking"],
                "description": "Should detect Digital Banking topic"
            },
            {
                "text": "Okash and Branch provide quick loans in Uganda",
                "expected_topics": ["Mobile Lending"],
                "description": "Should detect Mobile Lending topic"
            }
        ]
        
        for case in test_cases:
            topics, confidence = self.topic_extractor.extract_topics(case["text"])
            for expected_topic in case["expected_topics"]:
                assert expected_topic in topics, case["description"]
            assert confidence > 0.3, "Should have reasonable confidence"

    def test_vector_db_operations(self):
        """Test ChromaDB vector database operations"""
        # Test adding documents
        test_documents = [
            {
                "id": "test_1",
                "content": "MTN Mobile Money has high transaction fees",
                "source": "test",
                "post_id": 999,
                "topics": ["Mobile Money", "Fees"],
                "sentiment": "negative",
                "timestamp": "2023-11-01",
                "author": "test_user"
            }
        ]
        
        # Add to vector DB
        self.vector_db.add_documents(test_documents)
        
        # Test search
        results = self.vector_db.search_similar("mobile money fees", n_results=1)
        assert len(results) > 0
        assert "fees" in results[0]["content"].lower()
        
        # Test topic search
        topic_results = self.vector_db.search_by_topic("Mobile Money", n_results=1)
        assert len(topic_results) > 0
        
        # Test stats
        stats = self.vector_db.get_collection_stats()
        assert stats["total_documents"] >= 1
        
        # Clean up
        self.vector_db.delete_documents(["test_1"])