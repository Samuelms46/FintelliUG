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
    
    def detect_competitor_mentions(self, text):
        """Detect which competitors are mentioned in the text"""
        text_lower = text.lower()
        mentioned_competitors = []
        
        for competitor in self.competitors:
            # Simple keyword matching for competitor names
            competitor_keywords = competitor.lower().split()
            matches = sum(1 for keyword in competitor_keywords if keyword in text_lower)
            
            # If at least half the keywords match, consider it a mention
            if matches >= max(1, len(competitor_keywords) / 2):
                mentioned_competitors.append(competitor)
        
        return mentioned_competitors
    
    def analyze_competitor_sentiment(self, text, competitor):
        """Analyze sentiment toward a specific competitor"""
        try:
            prompt = f"""
            Analyze the sentiment toward {competitor} in this Ugandan fintech context.
            Text: {text}
            
            Return a JSON response with:
            - sentiment: positive, negative, or neutral
            - confidence: confidence score (0-1)
            - key_points: list of key points mentioned about the competitor
            
            Only return the JSON object, no other text.
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
                return {"sentiment": "neutral", "confidence": 0.5, "key_points": []}
                
        except Exception as e:
            print(f"Error in competitor sentiment analysis: {e}")
            return {"sentiment": "neutral", "confidence": 0.5, "key_points": []}
    
    def extract_competitor_insights(self, post_id, post_content):
        """Extract competitor insights from a post"""
        insights = []
        mentioned_competitors = self.detect_competitor_mentions(post_content)
        
        for competitor in mentioned_competitors:
            analysis = self.analyze_competitor_sentiment(post_content, competitor)
            
            insight_data = {
                "post_id": post_id,
                "competitor": competitor,
                "sentiment": analysis.get("sentiment", "neutral"),
                "context": post_content[:500],  # First 500 chars for context
                "extracted_insights": analysis.get("key_points", []),
                "confidence": analysis.get("confidence", 0.5)
            }
            
            # Save to database
            self.db_manager.add_competitor_mention(insight_data)
            insights.append(insight_data)
        
        return insights
    
    def _generate_summary_insights(self, insights):
        """Generate summary insights from competitor mentions"""
        if not insights:
            return []
        
        try:
            # If we have OpenAI API, use it for better insights
            if Config.OPENAI_API_KEY:
                insights_json = json.dumps(insights, default=str)
                
                prompt = f"""
                Analyze these competitor mentions from Uganda's fintech market and generate 3-5 summary insights.
                Focus on trends, sentiment shifts, and competitive positioning.
                
                Data: {insights_json}
                
                Return a JSON response with:
                - insights: list of insights, each with text, type, and confidence
                
                Only return the JSON object, no other text.
                """
                
                response = openai.ChatCompletion.create(
                    model=Config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.1
                )
                
                result = response.choices[0].message.content.strip()
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                    return result_data.get("insights", [])
            
            # Fallback: simple summary without OpenAI
            return self._generate_basic_summary(insights)
                
        except Exception as e:
            print(f"Error generating summary insights: {e}")
            return self._generate_basic_summary(insights)
    
    def _generate_basic_summary(self, insights):  # ADD HELPER METHOD
        """Generate basic summary without OpenAI"""
        if not insights:
            return []
        
        # Count sentiments by competitor
        competitor_stats = {}
        for insight in insights:
            competitor = insight.get('competitor', 'Unknown')
            sentiment = insight.get('sentiment', 'neutral')
            
            if competitor not in competitor_stats:
                competitor_stats[competitor] = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            competitor_stats[competitor][sentiment] += 1
        
        # Generate basic insights
        summary_insights = []
        
        # Insight 1: Overall sentiment leader
        if competitor_stats:
            best_competitor = max(competitor_stats.items(), 
                                key=lambda x: x[1]['positive'] - x[1]['negative'])
            summary_insights.append({
                "text": f"{best_competitor[0]} has the most positive sentiment overall",
                "type": "competitive_positioning",
                "confidence": 0.7
            })
        
        # Insight 2: Most discussed competitor
        most_discussed = max(competitor_stats.items(), key=lambda x: sum(x[1].values()))
        summary_insights.append({
            "text": f"{most_discussed[0]} is the most frequently discussed competitor",
            "type": "market_attention",
            "confidence": 0.8
        })
        
        # Insight 3: Sentiment trends
        negative_mentions = sum(1 for insight in insights if insight.get('sentiment') == 'negative')
        total_mentions = len(insights)
        if total_mentions > 0:
            negative_percent = (negative_mentions / total_mentions) * 100
            summary_insights.append({
                "text": f"{negative_percent:.1f}% of competitor mentions are negative",
                "type": "sentiment_trend",
                "confidence": 0.6
            })
        
        return summary_insights
    
    def generate_competitive_intelligence(self, hours=24):
        """Generate competitive intelligence report from recent data"""
        recent_posts = self.db_manager.get_recent_posts(hours=hours)
        competitor_insights = []
        
        for post in recent_posts:
            post_insights = self.extract_competitor_insights(post.id, post.cleaned_content)
            competitor_insights.extend(post_insights)
        
        # Generate summary insights
        summary = self._generate_summary_insights(competitor_insights)
        
        return {
            "time_period_hours": hours,
            "total_mentions": len(competitor_insights),
            "competitor_insights": competitor_insights,
            "summary": summary
        }