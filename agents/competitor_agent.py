import openai
import json
import re
from typing import Dict, List, Any
from config import Config
from database.db_manager import DatabaseManager
from utils.logger import app_logger

class CompetitorAnalysisAgent:
    def __init__(self):
        openai.api_key = Config.OPENAI_API_KEY
        self.competitors = Config.COMPETITORS
        self.db_manager = DatabaseManager()
        app_logger.info("CompetitorAnalysisAgent initialized")
    
    def find_competitor_mentions(self, hours: int = 24) -> List[Dict]:
        """Find posts mentioning competitors using vector database search"""
        competitor_mentions = []
        
        for competitor in self.competitors:
            # Search for posts mentioning this competitor
            results = self.db_manager.search_similar_posts(competitor, n_results=20)
            
            for result in results:
                # Check if this is a recent post (within specified hours)
                # You might need to add timestamp filtering logic here
                competitor_mentions.append({
                    "post_id": result["metadata"].get("post_id"),
                    "competitor": competitor,
                    "content": result["content"],
                    "source": result["metadata"].get("source"),
                    "timestamp": result["metadata"].get("timestamp")
                })
        
        return competitor_mentions
    
    def generate_competitive_intelligence(self, hours=24):
        """Generate competitive intelligence report using vector DB"""
        # Find competitor mentions using vector search
        competitor_mentions = self.find_competitor_mentions(hours)
        
        # Analyze each mention
        analyzed_mentions = []
        for mention in competitor_mentions:
            analysis = self.analyze_competitor_sentiment(mention["content"], mention["competitor"])
            
            insight_data = {
                "post_id": mention["post_id"],
                "competitor": mention["competitor"],
                "sentiment": analysis.get("sentiment", "neutral"),
                "context": mention["content"][:500],
                "extracted_insights": analysis.get("key_points", []),
                "confidence": analysis.get("confidence", 0.5)
            }
            
            # Save to database
            self.db_manager.add_competitor_mention(insight_data)
            analyzed_mentions.append(insight_data)
        
        # Generate summary insights
        summary = self._generate_summary_insights(analyzed_mentions)
        
        return {
            "time_period_hours": hours,
            "total_mentions": len(analyzed_mentions),
            "competitor_mentions": analyzed_mentions,
            "summary": summary
        }