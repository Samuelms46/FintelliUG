import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.competitor_agent import CompetitorAnalysisAgent
from agents.coordinator import CoordinatorAgent
from database.db_manager import DatabaseManager
from config import Config

class TestAgents:
    def setup_method(self):
        self.competitor_agent = CompetitorAnalysisAgent()
        self.coordinator_agent = CoordinatorAgent()
        self.db_manager = DatabaseManager()
        
    def test_competitor_detection_uganda_fintech(self):
        """Test competitor detection for Uganda fintech market"""
        test_cases = [
            {
                "text": "MTN Mobile Money has the best network coverage",
                "expected_competitors": ["MTN Mobile Money"],
                "description": "Should detect MTN Mobile Money"
            },
            {
                "text": "Airtel Money is better for sending money to villages",
                "expected_competitors": ["Airtel Money"],
                "description": "Should detect Airtel Money"
            },
            {
                "text": "Chipper Cash is expanding to Uganda with lower fees",
                "expected_competitors": ["Chipper Cash"],
                "description": "Should detect Chipper Cash"
            },
            {
                "text": "I use both MTN and Airtel for different purposes",
                "expected_competitors": ["MTN Mobile Money", "Airtel Money"],
                "description": "Should detect multiple competitors"
            }
        ]
        
        for case in test_cases:
            detected = self.competitor_agent.detect_competitor_mentions(case["text"])
            for expected in case["expected_competitors"]:
                assert expected in detected, case["description"]
    
    def test_insight_synthesis_quality(self):
        """Test that insight synthesis produces high-quality results"""
        # Mock realistic insights from different agents
        social_insights = [
            {
                "type": "social_trend",
                "content": "50% increase in discussions about mobile money fees",
                "confidence": 0.8,
                "topic": "Fees"
            }
        ]
        
        competitor_insights = [
            {
                "type": "competitor_analysis",
                "content": "MTN Mobile Money receiving negative sentiment due to fee increases",
                "confidence": 0.7,
                "competitor": "MTN Mobile Money",
                "sentiment": "negative"
            }
        ]
        
        sentiment_insights = [
            {
                "type": "market_sentiment", 
                "content": "Overall market sentiment declining due to fee concerns",
                "confidence": 0.6,
                "score": 6.2
            }
        ]
        
        # Test synthesis
        synthesized = self.coordinator_agent.synthesize_insights(
            social_insights, competitor_insights, sentiment_insights
        )
        
        # Verify structure and quality
        required_keys = ["executive_summary", "market_health_score", "investment_opportunities", "risks", "recommendations"]
        for key in required_keys:
            assert key in synthesized, f"Synthesized insights should contain {key}"
        
        assert 0 <= synthesized["market_health_score"] <= 10, "Market health score should be between 0-10"
        assert synthesized["confidence"] >= 0.5, "Should have reasonable confidence"
        
        # Verify insights are actionable
        assert len(synthesized["recommendations"]) > 0, "Should provide actionable recommendations"
        assert len(synthesized["investment_opportunities"]) > 0, "Should identify investment opportunities"