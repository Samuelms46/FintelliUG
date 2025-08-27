import openai
import json
import re
from typing import Dict, List, Any
from database.db_manager import DatabaseManager
from config import Config
from utils.logger import app_logger

class CoordinatorAgent:
    def __init__(self):
        openai.api_key = Config.OPENAI_API_KEY
        self.db_manager = DatabaseManager()
        app_logger.info("CoordinatorAgent initialized")
    
    def synthesize_insights(self, social_insights: List[Dict], competitor_insights: List[Dict], 
                           sentiment_insights: List[Dict]) -> Dict:
        """
        Synthesize insights from all specialized agents into a comprehensive report
        """
        # If we have real data, use it. Otherwise, use the fallback insights
        has_real_data = any([
            social_insights and len(social_insights) > 0,
            competitor_insights and len(competitor_insights) > 0,
            sentiment_insights and len(sentiment_insights) > 0
        ])
        if not has_real_data:
            app_logger.warning("No real data available. Using fallback insights.")
            return self._generate_fallback_insights(social_insights, competitor_insights, sentiment_insights)
        
        # If we have real data, synthesize insights
        try:
            # Prepare data for synthesis
            insights_data = {
                "social_insights": social_insights,
                "competitor_insights": competitor_insights,
                "sentiment_insights": sentiment_insights
            }
            
            # Count total insights for Confidence calculation
            total_insights = (len(social_insights or [])) +
                            len(competitor_insights or []) + 
                            len(sentiment_insights or []))
            base_confidence = min(0.3 + (total_insights * 0.02), 0.9) # social confidence with hight volume


            prompt =  f"""
            As the Chief Intelligence Officer for Uganda's fintech market, analyze these insights 
            and create a comprehensive intelligence report.
            
            DATA FROM AGENTS:
            {json.dumps(insights_data, indent=2)}
            
            CONTEXT: Uganda fintech market, mobile money, digital banking, financial services.
            
            Create a JSON report with:
            - executive_summary: 2-3 sentence overview of the most important findings
            - key_trends: 3-5 major trends with brief evidence from the data
            - market_health_score: 1-10 score based on sentiment and activity
            - investment_opportunities: 2-3 specific opportunities with potential (High/Medium/Low)
            - risks: 2-3 specific risks with severity (High/Medium/Low) 
            - recommendations: 3-5 actionable recommendations
            - confidence: 0-1 score based on data quality and quantity
            
            Focus on Uganda-specific context and make insights actionable for investors.
            """
            
            response = openai.ChatCompletion.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1 # Lower temp for a more consistent results
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                synthesized_insights = json.loads(json_match.group())
                 # Enhance with real data metrics
                synthesized_insights['data_metrics'] = {
                    'social_insights_count': len(social_insights or []),
                    'competitor_insights_count': len(competitor_insights or []),
                    'sentiment_insights_count': len(sentiment_insights or [])
                }
                app_logger.info("Successfully synthesized insights from all agents")
                return synthesized_insights
            else:
                app_logger.error("Failed to parse JSON from coordinator agent response")
                return self._generate_fallback_insights(social_insights, competitor_insights, sentiment_insights)
                
        except Exception as e:
            app_logger.error(f"Error in coordinator agent: {e}")
                        return self._generate_data_driven_insights({
                "social_insights": social_insights or [],
                "competitor_insights": competitor_insights or [], 
                "sentiment_insights": sentiment_insights or []
            })



    def _generate_data_driven_insights(self, insights_data: Dict) -> Dict:
        """Generate insights directly from data without OpenAI"""
        social_insights = insights_data.get('social_insights', [])
        competitor_insights = insights_data.get('competitor_insights', [])
        sentiment_insights = insights_data.get('sentiment_insights', [])
        
        # Calculate metrics from real data
        total_insights = len(social_insights) + len(competitor_insights) + len(sentiment_insights)
        
        # Analyze competitor sentiment
        competitor_sentiment = {}
        for insight in competitor_insights:
            competitor = insight.get('competitor', 'Unknown')
            sentiment = insight.get('sentiment', 'neutral')
            if competitor not in competitor_sentiment:
                competitor_sentiment[competitor] = {'positive': 0, 'negative': 0, 'neutral': 0}
            competitor_sentiment[competitor][sentiment] += 1
        
        # Generate insights based on actual data
        trends = []
        opportunities = []
        risks = []
        
        # Add trends based on actual data
        if competitor_insights:
            trends.append("Competitor discussions are active in the market")
        if sentiment_insights:
            positive_count = sum(1 for i in sentiment_insights if 'positive' in str(i).lower())
            if positive_count > len(sentiment_insights) / 2:
                trends.append("Overall positive sentiment in recent discussions")
        
        # Add opportunities based on data
        if competitor_insights:
            opportunities.append({"opportunity": "Competitive intelligence gathering", "potential": "High"})
        
        # Add risks based on data
        if any('negative' in str(i).lower() for i in competitor_insights):
            risks.append({"risk": "Negative competitor sentiment", "severity": "Medium"})
        
        return {
            "executive_summary": f"Analysis based on {total_insights} data points from social media monitoring",
            "key_trends": trends or ["Market activity detected in social discussions"],
            "market_health_score": 7.0,
            "investment_opportunities": opportunities or [{"opportunity": "Market monitoring", "potential": "Medium"}],
            "risks": risks or [{"risk": "Limited data availability", "severity": "Low"}],
            "recommendations": [
                "Increase data collection for better insights",
                "Monitor competitor activities closely",
                "Track customer sentiment trends"
            ],
            "confidence": min(0.3 + (total_insights * 0.05), 0.8),
            "data_metrics": {
                'total_insights': total_insights,
                'source': 'data_driven_analysis'
            }
        }
    
    def _generate_fallback_insights(self):
        """Fallback when no data is available"""
        return {
            "executive_summary": "Initial system analysis - awaiting more data collection",
            "key_trends": [
                "Uganda fintech market shows growth potential",
                "Mobile money adoption increasing across the country",
                "Regulatory environment evolving"
            ],
            "market_health_score": 6.5,
            "investment_opportunities": [
                {"opportunity": "Digital financial inclusion", "potential": "High"},
                {"opportunity": Mobile payment solutions", "potential": "Medium"}
            ],
            "risks": [
                {"risk": "Regulatory uncertainty", "severity": "Medium"},
                {"risk": "Infrastructure challenges", "severity": "Medium"}
            ],
            "recommendations": [
                "Deploy more data collection sources",
                "Establish baseline metrics",
                "Develop competitor monitoring framework"
            ],
            "confidence": 0.4,
            "data_metrics": {
                'total_insights': 0,
                'source': 'fallback_analysis'
            }
        }
    



    
    def _generate_fallback_insights(self, social_insights, competitor_insights, sentiment_insights):
        """Generate fallback insights when LLM fails"""
        return {
            "executive_summary": "Synthesized analysis of Uganda fintech market trends",
            "key_trends": [
                "Growing customer discussions around mobile money fees and services",
                "Increased regulatory attention affecting market dynamics",
                "Competition intensifying between major players"
            ],
            "market_health_score": 7.5,
            "investment_opportunities": [
                {"opportunity": "Rural mobile money expansion", "potential": "High"},
                {"opportunity": "Cross-border payment solutions", "potential": "Medium"},
                {"opportunity": "Digital lending platforms", "potential": "Medium"}
            ],
            "risks": [
                {"risk": "Regulatory changes", "severity": "High"},
                {"risk": "Customer dissatisfaction with fees", "severity": "Medium"},
                {"risk": "Market saturation in urban areas", "severity": "Medium"}
            ],
            "recommendations": [
                "Monitor regulatory developments closely",
                "Focus on customer experience improvements",
                "Explore underserved rural markets"
            ],
            "confidence": 0.7
        }
    
    def resolve_conflicts(self, agent_insights: Dict[str, List[Dict]]) -> Dict:
        """
        Resolve conflicts between different agent insights
        """
        conflicts = self._identify_conflicts(agent_insights)
        
        if not conflicts:
            return {"resolved": True, "conflicts": [], "resolution": "No conflicts detected"}
        
        try:
            prompt = f"""
            As an impartial arbitrator, resolve these conflicting insights about Uganda's fintech market:
            
            Conflicts: {json.dumps(conflicts, indent=2)}
            
            For each conflict, analyze the evidence and provide a resolution.
            Consider: data quality, source reliability, temporal relevance, and contextual factors.
            
            Return a JSON response with:
            - resolved: boolean indicating if conflicts were resolved
            - conflicts: list of original conflicts
            - resolutions: list of resolutions for each conflict
            - final_judgment: overall assessment of which perspective is more reliable
            
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
                resolution = json.loads(json_match.group())
                app_logger.info(f"Resolved {len(conflicts)} conflicts between agents")
                return resolution
            else:
                return {"resolved": False, "conflicts": conflicts, "resolution": "Failed to resolve conflicts"}
                
        except Exception as e:
            app_logger.error(f"Error in conflict resolution: {e}")
            return {"resolved": False, "conflicts": conflicts, "resolution": f"Error: {str(e)}"}
    
    def _identify_conflicts(self, agent_insights: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Identify conflicts between different agent insights
        """
        conflicts = []
        
        # Extract all insights by type and topic
        insights_by_topic = {}
        for agent_type, insights in agent_insights.items():
            for insight in insights:
                topic = insight.get('topic', 'general')
                if topic not in insights_by_topic:
                    insights_by_topic[topic] = []
                insights_by_topic[topic].append({
                    'agent': agent_type,
                    'insight': insight,
                    'confidence': insight.get('confidence', 0.5)
                })
        
        # Look for conflicts within each topic
        for topic, topic_insights in insights_by_topic.items():
            if len(topic_insights) < 2:
                continue
                
            # Check for sentiment conflicts
            sentiments = {}
            for insight_data in topic_insights:
                insight = insight_data['insight']
                if 'sentiment' in insight:
                    agent = insight_data['agent']
                    sentiment = insight['sentiment']
                    if sentiment not in sentiments:
                        sentiments[sentiment] = []
                    sentiments[sentiment].append(agent)
            
            # If multiple sentiments detected for same topic, it's a conflict
            if len(sentiments) > 1:
                conflicts.append({
                    'topic': topic,
                    'type': 'sentiment_conflict',
                    'evidence': sentiments,
                    'description': f'Conflicting sentiments about {topic}: {sentiments}'
                })
        
        return conflicts
    
    def generate_daily_briefing(self, synthesized_insights: Dict) -> Dict:
        """
        Generate a daily briefing document for investors
        """
        try:
            prompt = f"""
            Create a professional daily briefing for fintech investors focused on Uganda.
            
            Synthesized insights: {json.dumps(synthesized_insights, indent=2)}
            
            Format the briefing with:
            1. Executive Summary (3-4 sentences)
            2. Market Health Score and Trend
            3. Key Developments (bullet points)
            4. Competitive Landscape Update
            5. Investment Opportunities (ranked)
            6. Risk Assessment
            7. Recommended Actions
            
            Return a JSON response with:
            - title: Briefing title with date
            - executive_summary: 3-4 sentence overview
            - sections: List of sections with titles and content
            - key_takeaways: List of 3-5 key takeaways
            - confidence: Overall confidence (0-1)
            
            Only return the JSON object, no other text.
            """
            
            response = openai.ChatCompletion.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                briefing = json.loads(json_match.group())
                
                # Store briefing in database
                self.db_manager.add_insight({
                    "type": "daily_briefing",
                    "content": json.dumps(briefing),
                    "confidence": briefing.get('confidence', 0.8),
                    "source_data": []  # All insights contribute to briefing
                })
                
                app_logger.info("Generated daily briefing for investors")
                return briefing
            else:
                return self._generate_fallback_briefing(synthesized_insights)
                
        except Exception as e:
            app_logger.error(f"Error generating daily briefing: {e}")
            return self._generate_fallback_briefing(synthesized_insights)
    
    def _generate_fallback_briefing(self, insights: Dict) -> Dict:
        """Generate fallback briefing when LLM fails"""
        from datetime import datetime
        
        return {
            "title": f"Uganda Fintech Daily Briefing - {datetime.now().strftime('%Y-%m-%d')}",
            "executive_summary": "Comprehensive analysis of Uganda's fintech market trends, opportunities, and risks based on social media intelligence.",
            "sections": [
                {
                    "title": "Market Overview",
                    "content": "The Uganda fintech market shows steady growth with increasing mobile money adoption and competitive dynamics."
                },
                {
                    "title": "Key Developments",
                    "content": "- Customer discussions focus on service fees and reliability\n- Regulatory developments creating market uncertainty\n- New competitive threats emerging"
                },
                {
                    "title": "Investment Opportunities",
                    "content": "1. Rural financial inclusion initiatives\n2. Cross-border payment solutions\n3. Digital lending platforms"
                }
            ],
            "key_takeaways": [
                "Market health score: 7.5/10",
                "Regulatory changes represent the biggest risk",
                "Rural markets offer significant growth potential"
            ],
            "confidence": 0.7
        }