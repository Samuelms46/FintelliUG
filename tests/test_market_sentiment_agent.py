import pytest
from agents.market_sentiment_agent import MarketSentimentAgent
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_market_sentiment_agent_initialization():
    """Test that the agent initializes correctly."""
    agent = MarketSentimentAgent()
    assert agent.name == "market_sentiment"
    assert len(agent.market_segments) > 0
    assert len(agent.risk_factors) > 0

def test_market_sentiment_agent_validation():
    """Test input validation."""
    agent = MarketSentimentAgent()
    
    # Valid inputs
    assert agent.validate_input({"days": 7})
    assert agent.validate_input({"hours": 24})
    assert agent.validate_input({"segment": "mobile_money"})
    assert agent.validate_input({"posts": [{"text": "Test", "timestamp": datetime.now().isoformat()}]})
    
    # Invalid inputs
    assert not agent.validate_input("not a dict")
    assert not agent.validate_input({})
    assert not agent.validate_input({"segment": "invalid_segment"})

def test_market_sentiment_agent_with_mock_data():
    """Test the agent with mock data."""
    agent = MarketSentimentAgent()
    
    # Create mock posts
    mock_posts = [
        {
            "text": "MTN Mobile Money fees are getting too high. Thinking of switching to Airtel.",
            "source": "twitter",
            "timestamp": datetime.now().isoformat(),
            "sentiment": "negative",
            "sentiment_score": 0.2
        },
        {
            "text": "Loving the new Chipper Cash app update. Much easier to send money across borders now!",
            "source": "reddit",
            "timestamp": datetime.now().isoformat(),
            "sentiment": "positive",
            "sentiment_score": 0.8
        },
        {
            "text": "Bank of Uganda announces new regulations for mobile money providers.",
            "source": "news",
            "timestamp": datetime.now().isoformat(),
            "sentiment": "neutral",
            "sentiment_score": 0.5
        }
    ]
    
    # Process the mock data
    result = agent.process({"posts": mock_posts})
    
    # Verify the result structure
    assert "market_sentiment" in result
    assert "segment_trends" in result
    assert "investment_opportunities" in result
    assert "market_risks" in result
    assert "health_indicators" in result
    
    # Verify market sentiment
    assert "overall_sentiment" in result["market_sentiment"]
    assert "sentiment_score" in result["market_sentiment"]
    
    # Verify health indicators
    assert "market_health" in result["health_indicators"]
    assert "health_score" in result["health_indicators"]
    assert "opportunity_score" in result["health_indicators"]
