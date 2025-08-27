import os
import sys
from typing import Dict, Any
from unittest.mock import patch

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.social_intel_agent import SocialIntelAgent

def test_query_based_processing_mock():
    """Test the agent with mocked social data fetching."""
    print("=== Testing Query-Based Social Intelligence Processing (Mock Data) ===\n")
    
    # Mock data that would come from X API
    mock_social_data = [
        {
            "text": "Just tried Chipper Cash for international money transfer from Uganda to Kenya. Fast and affordable! #fintech #ChipperCash",
            "source": "twitter",
            "timestamp": "2025-08-26T10:00:00Z"
        },
        {
            "text": "Chipper Cash is revolutionizing digital payments in East Africa. Great user experience for mobile money transfers.",
            "source": "twitter",
            "timestamp": "2025-08-26T11:30:00Z"
        },
        {
            "text": "Love how Chipper Cash makes cross-border payments so simple. The future of fintech in Uganda! 🚀",
            "source": "twitter",
            "timestamp": "2025-08-26T12:15:00Z"
        }
    ]
    
    try:
        # Initialize the agent
        agent = SocialIntelAgent(config={})
        
        # Mock the fetch_social_data method to return our mock data
        with patch.object(agent, 'fetch_social_data', return_value=mock_social_data):
            # Test input with query for dynamic social data fetching
            test_input = {
                "query": "Chipper Cash",
                "max_results": 15
            }
            
            print(f"Input: {test_input}")
            print("Processing social intelligence with mock social data...\n")
            
            # Process the query
            result = agent.process(test_input)
            
            # Display results
            if result.get('error'):
                print(f"Error: {result['error']}")
            else:
                print("=== RESULTS ===")
                print(f"Agent: {result['agent']}")
                print(f"Query: {result.get('query_info', 'N/A')}")
                print(f"Posts Processed: {result.get('posts_processed', 0)}")
                print(f"Relevant Posts: {result.get('relevant_posts', 0)}")
                print(f"Data Quality Score: {result.get('data_quality_score', 0):.2f}")
                
                print(f"\nSentiment Analysis:")
                sentiment = result.get('sentiment_analysis', {})
                print(f"  Overall: {sentiment.get('overall_sentiment', 'N/A')}")
                print(f"  Score: {sentiment.get('sentiment_score', 'N/A')}")
                
                print(f"\nTrending Topics:")
                for topic in result.get('trending_topics', [])[:3]:
                    print(f"  - {topic['topic']}: {topic['mention_count']} mentions")
                
                print(f"\nInsights Generated: {len(result.get('insights', []))}")
                for i, insight in enumerate(result.get('insights', [])[:3], 1):
                    print(f"  {i}. [{insight['type']}] {insight['insight']}")
                    print(f"     Confidence: {insight.get('confidence', 0):.2f}")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

def test_traditional_posts_processing():
    """Test the agent with traditional posts input."""
    print("\n=== Testing Traditional Posts Processing ===\n")
    
    try:
        # Initialize the agent
        agent = SocialIntelAgent(config={})
        
        # Test input with sample posts
        test_input = {
            "posts": [
                {
                    "text": "MTN MoMo service is down again. This is affecting my business transactions. Need better reliability! #fintech #Uganda",
                    "source": "twitter",
                    "timestamp": "2025-08-26T10:00:00Z"
                },
                {
                    "text": "Airtel Money has improved a lot this year. Faster transactions and better customer service. #digitalpayments",
                    "source": "twitter", 
                    "timestamp": "2025-08-26T11:30:00Z"
                },
                {
                    "text": "The future of banking in Uganda is mobile-first. Traditional banks need to adapt quickly or get left behind.",
                    "source": "twitter",
                    "timestamp": "2025-08-26T12:15:00Z"
                },
                {
                    "text": "Just got a loan through my mobile banking app. The digital transformation in Uganda's financial sector is amazing!",
                    "source": "twitter",
                    "timestamp": "2025-08-26T13:00:00Z"
                }
            ]
        }
        
        print(f"Processing {len(test_input['posts'])} sample posts...")
        
        # Process the posts
        result = agent.process(test_input)
        
        # Display results
        if result.get('error'):
            print(f"Error: {result['error']}")
        else:
            print("=== RESULTS ===")
            print(f"Posts Processed: {result.get('posts_processed', 0)}")
            print(f"Relevant Posts: {result.get('relevant_posts', 0)}")
            print(f"Data Quality Score: {result.get('data_quality_score', 0):.2f}")
            
            print(f"\nSentiment Analysis:")
            sentiment = result.get('sentiment_analysis', {})
            print(f"  Overall: {sentiment.get('overall_sentiment', 'N/A')}")
            
            print(f"\nInsights Generated: {len(result.get('insights', []))}")
            for insight in result.get('insights', []):
                print(f"  - {insight['insight']}")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    print("Enhanced SocialIntelAgent Test Suite (With Mock Data)")
    print("=" * 60)
    
    # Test both input methods
    test_query_based_processing_mock()
    test_traditional_posts_processing()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
