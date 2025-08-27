"""
Test script demonstrating the enhanced SocialIntelAgent with XSearchTool integration.
"""

import os
import sys
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.social_intel_agent import SocialIntelAgent

def test_query_based_processing():
    """Test the agent with query-based social data fetching."""
    print("=== Testing Query-Based Social Intelligence Processing ===\n")
    
    try:
        # Initialize the agent
        agent = SocialIntelAgent(config={})
        
        # Test input with query for dynamic social data fetching
        test_input = {
            "query": "mobile money Uganda",
            "max_results": 15
        }
        
        print(f"Input: {test_input}")
        print("Processing social intelligence with dynamic data fetching...\n")
        
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
                    "text": "Just used MTN MoMo to pay for groceries. So convenient! #fintech #Uganda",
                    "source": "twitter",
                    "timestamp": "2025-08-26T10:00:00Z"
                },
                {
                    "text": "Airtel Money transaction failed again. This is frustrating! Need better digital payment options.",
                    "source": "twitter", 
                    "timestamp": "2025-08-26T11:30:00Z"
                },
                {
                    "text": "The future of banking in Uganda is digital. Mobile money adoption is growing rapidly.",
                    "source": "twitter",
                    "timestamp": "2025-08-26T12:15:00Z"
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
            print(f"Insights Generated: {len(result.get('insights', []))}")
            
            for insight in result.get('insights', []):
                print(f"  - {insight['insight']}")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    print("Enhanced SocialIntelAgent Test Suite")
    print("=" * 50)
    
    # Test both input methods
    test_query_based_processing()
    test_traditional_posts_processing()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
