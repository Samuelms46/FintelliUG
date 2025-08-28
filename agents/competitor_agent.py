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
    
    def extract_competitor_insights(self, post_id: int, content: str) -> List[Dict]:
        """Extract competitor insights from a single post"""
        insights = []
        
        for competitor in self.competitors:
            if competitor.lower() in content.lower():
                analysis = self.analyze_competitor_sentiment(content, competitor)
                
                insight = {
                    "post_id": post_id,
                    "competitor": competitor,
                    "sentiment": analysis.get("sentiment", "neutral"),
                    "context": content[:500],
                    "extracted_insights": analysis.get("key_points", []),
                    "confidence": analysis.get("confidence", 0.5)
                }
                
                # Save to database
                self.db_manager.add_competitor_mention(insight)
                insights.append(insight)
        
        return insights
    
    def analyze_competitor_sentiment(self, content: str, competitor: str) -> Dict:
        """Analyze sentiment and extract insights for a specific competitor"""
        try:
            prompt = f"""
            Analyze this content mentioning {competitor} in Uganda's fintech market:
            
            Content: {content}
            
            Return a JSON object with:
            - sentiment: "positive", "negative", or "neutral"
            - key_points: list of key insights extracted
            - confidence: confidence score (0-1)
            
            Focus on customer sentiment, product mentions, pricing, and competitive positioning.
            """
            
            response = openai.ChatCompletion.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "sentiment": "neutral",
                    "key_points": [f"Mention of {competitor}"],
                    "confidence": 0.5
                }
                
        except Exception as e:
            app_logger.error(f"Error analyzing competitor sentiment: {e}")
            return {
                "sentiment": "neutral",
                "key_points": [f"Mention of {competitor}"],
                "confidence": 0.3
            }
    
    def _generate_summary_insights(self, analyzed_mentions: List[Dict]) -> List[Dict]:
        """Generate summary insights from analyzed competitor mentions"""
        if not analyzed_mentions:
            return []
        
        try:
            # Group by competitor
            competitor_data = {}
            for mention in analyzed_mentions:
                competitor = mention["competitor"]
                if competitor not in competitor_data:
                    competitor_data[competitor] = {
                        "mentions": [],
                        "sentiments": {"positive": 0, "negative": 0, "neutral": 0}
                    }
                competitor_data[competitor]["mentions"].append(mention)
                sentiment = mention.get("sentiment", "neutral")
                competitor_data[competitor]["sentiments"][sentiment] += 1
            
            # Generate insights for each competitor
            summary_insights = []
            for competitor, data in competitor_data.items():
                total_mentions = len(data["mentions"])
                positive_pct = (data["sentiments"]["positive"] / total_mentions) * 100
                negative_pct = (data["sentiments"]["negative"] / total_mentions) * 100
                
                # Determine overall sentiment
                if positive_pct > negative_pct:
                    overall_sentiment = "positive"
                    confidence = positive_pct / 100
                elif negative_pct > positive_pct:
                    overall_sentiment = "negative"
                    confidence = negative_pct / 100
                else:
                    overall_sentiment = "neutral"
                    confidence = 0.5
                
                summary_insights.append({
                    "competitor": competitor,
                    "text": f"{competitor}: {overall_sentiment} sentiment ({positive_pct:.1f}% positive, {negative_pct:.1f}% negative)",
                    "confidence": confidence,
                    "total_mentions": total_mentions
                })
            
            return summary_insights
            
        except Exception as e:
            app_logger.error(f"Error generating summary insights: {e}")
            return []
    
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