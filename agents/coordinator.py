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
    
    def synthesize_insights(self,
                            social_insights: List[Dict[str, Any]], 
                            competitor_mentions: List[Dict[str, Any]], 
                           market_insights: List[Dict[str, Any]],
                            market_analysis: Dict[str, Any] = None):
        """Synthesize insights from multiple agents into a coherent analysis."""
        try:
            # Prepare data for synthesis
            insights_data = {
                "social_insights": social_insights,
                "competitor_mentions": competitor_mentions,
                "market_insights": market_insights
            }
            
            prompt = f"""
            As the Chief Intelligence Officer for Uganda's fintech market, synthesize these insights 
            from specialized agents into a comprehensive intelligence report.
            
            Data from agents:
            {json.dumps(insights_data, indent=2)}
            
            Instructions:
            1. Identify overarching trends and patterns
            2. Resolve any conflicts between different agent perspectives
            3. Highlight the 3-5 most important findings
            4. Provide actionable recommendations for investors
            5. Assess market health on a scale of 1-10
            6. Identify potential risks and opportunities
            
            Return a JSON response with:
            - executive_summary: Brief overview of key findings
            - key_trends: List of major trends with evidence
            - market_health_score: Overall market health (1-10)
            - investment_opportunities: Ranked list of opportunities
            - risks: List of potential risks with severity
            - recommendations: Actionable recommendations
            - confidence: Overall confidence in analysis (0-1)
            
            Only return the JSON object, no other text.
            """
            
            response = openai.ChatCompletion.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                synthesized = json.loads(json_match.group())

                # Add market analysis if available
                if market_analysis and "health_indicators" in market_analysis:
                    health = market_analysis["health_indicators"]
                    synthesized["market_health"] = health.get("market_health", "unknown")
                    synthesized["opportunity_score"] = health.get("opportunity_score", 0.5)
                    synthesized["risk_level"] = health.get("risk_level", 0.5)
                    synthesized["growth_segments"] = health.get("growth_segments", [])

                # Add investment opportunities if available
                if market_analysis and "investment_opportunities" in market_analysis:
                    synthesized["investment_opportunities"] = market_analysis["investment_opportunities"]

                app_logger.info("Successfully synthesized insights from all agents")
                return synthesized
            else:
                app_logger.error("Failed to parse JSON from coordinator agent response")
                return self._generate_fallback_insights(social_insights, competitor_mentions, market_insights)
                
        except Exception as e:
            app_logger.error(f"Error in coordinator agent: {e}")
            return self._generate_fallback_insights(social_insights, competitor_mentions, market_insights)
    
    def _generate_fallback_insights(self, social_insights, competitor_mentions, market_insights):
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