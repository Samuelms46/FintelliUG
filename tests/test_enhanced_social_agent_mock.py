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
        "text": "Just used MTN MoMo to pay my electricity bill. The transaction fee went up to 1,500 UGX! When did they increase their fees? #Uganda #MobileMoney",
        "source": "twitter",
        "timestamp": "2025-08-30T09:15:00Z",
        "relevance_score": 0.85
    },
    {
        "text": "Airtel Money's new cashback promotion is amazing! Got 2% back on my last three transactions. Much better deal than MTN right now. #DigitalPayments #Uganda",
        "source": "twitter",
        "timestamp": "2025-08-30T10:23:00Z",
        "relevance_score": 0.9
    },
    {
        "text": "The MTN MoMo app keeps crashing when I try to send money to my family in Mbale. @MTNUganda please fix this! Been trying for 2 days now. #CustomerService #MobileMoney",
        "source": "twitter",
        "timestamp": "2025-08-30T11:45:00Z",
        "relevance_score": 0.8
    },
    {
        "text": "Comparing fees between Airtel Money and MTN MoMo for sending 100,000 UGX to Jinja. Airtel charges 1,800 while MTN charges 2,200. The difference adds up! #FinancialLiteracy #Uganda",
        "source": "twitter",
        "timestamp": "2025-08-30T13:10:00Z",
        "relevance_score": 0.95
    },
    {
        "text": "Bank transfers still taking 24 hours while mobile money is instant. This is why fintech is winning in Uganda. Just sent money via Airtel Money and my supplier received it immediately. #BusinessTips",
        "source": "twitter",
        "timestamp": "2025-08-30T14:30:00Z",
        "relevance_score": 0.85
    },
    {
        "text": "MTN MoMo agents in rural areas charging extra 'service fees' on top of the official rates. Is this legal? Anyone else experiencing this in Karamoja region? #ConsumerRights #Uganda",
        "source": "twitter",
        "timestamp": "2025-08-30T15:45:00Z",
        "relevance_score": 0.9
    },
    {
        "text": "Tried Chipper Cash for the first time to send money to Kenya. Lower fees than both MTN and Airtel for international transfers! Game changer for cross-border business. #EastAfrica #Fintech",
        "source": "twitter",
        "timestamp": "2025-08-30T16:20:00Z",
        "relevance_score": 0.85
    },
    {
        "text": "The new Airtel Money savings feature giving 5% interest is better than most bank savings accounts in Uganda right now. Finally some competition in the financial sector! #FinancialInclusion",
        "source": "twitter",
        "timestamp": "2025-08-30T17:05:00Z",
        "relevance_score": 0.8
    },
    {
        "text": "MTN MoMo needs to improve their customer service. Been waiting for 3 hours for them to reverse a wrong transaction. Airtel resolves such issues in minutes. #CustomerExperience #Uganda",
        "source": "twitter",
        "timestamp": "2025-08-30T18:30:00Z",
        "relevance_score": 0.9
    },
    {
        "text": "Mobile money agents in Kampala running out of float by afternoon. This happens every month-end. Both MTN and Airtel need to solve this liquidity problem. #UrbanChallenges #DigitalPayments",
        "source": "twitter",
        "timestamp": "2025-08-30T19:15:00Z",
        "relevance_score": 0.85
    },
    {
        "text": "Just got a micro-loan through MTN MoMo MoKash in 2 minutes! No paperwork, no collateral. This is how fintech is changing financial access in Uganda. #FinancialInclusion #Fintech",
        "source": "twitter",
        "timestamp": "2025-08-31T08:10:00Z",
        "relevance_score": 0.9
    },
    {
        "text": "Airtel Money's market share growing in Western Uganda. Their agent network has expanded significantly in the last 6 months. MTN needs to step up their game. #Competition #MobileMoney",
        "source": "twitter",
        "timestamp": "2025-08-31T09:25:00Z",
        "relevance_score": 0.85
    },
    {
        "text": "The URA tax on mobile money transactions is hurting small businesses. Paid 7,500 UGX in fees+tax for a 300K transaction via MTN MoMo yesterday. Too expensive! #TaxPolicy #Uganda",
        "source": "twitter",
        "timestamp": "2025-08-31T10:40:00Z",
        "relevance_score": 0.95
    },
    {
        "text": "Security alert: Beware of scammers calling and pretending to be Airtel Money customer care. They tried to get my PIN today. Always verify before sharing any info! #CyberSecurity #Uganda",
        "source": "twitter",
        "timestamp": "2025-08-31T11:55:00Z",
        "relevance_score": 0.8
    },
    {
        "text": "MTN MoMo's integration with utility companies makes bill payment so convenient. Just paid my NWSC and Umeme bills in under 2 minutes. #DigitalTransformation #Uganda",
        "source": "twitter",
        "timestamp": "2025-08-31T13:05:00Z",
        "relevance_score": 0.85
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
