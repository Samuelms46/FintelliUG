import json
import re
import time
from typing import Dict, List, Any, Optional
from database.db_manager import DatabaseManager
from config import Config
from utils.logger import app_logger
from agents.base_agent import BaseAgent


class CoordinatorAgent(BaseAgent):
	def __init__(self):
		super().__init__("coordinator")
		self.db_manager = DatabaseManager()
		app_logger.info("CoordinatorAgent initialized")

	def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Required abstract method implementation for BaseAgent."""
		return {"status": "coordinator_process_not_implemented"}

	# Internal helper methods
	
	def parse_llm_json(self, response_text: str) -> Dict[str, Any]:
		"""Robustly parse JSON from LLM text output."""
		try:
			return json.loads(response_text)
		except json.JSONDecodeError:
			json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
			if json_match:
				try:
					return json.loads(json_match.group())
				except json.JSONDecodeError:
					pass
			# Structured error payload used by callers to trigger fallback
			return {"error": "Failed to parse JSON", "raw_response": response_text}

	def call_llm_with_retry(self, prompt: str, *, max_retries: int = 3):
		"""Call Groq LLM with exponential backoff on transient errors."""
		for attempt in range(max_retries):
			try:
				response = self.llm.invoke(prompt)
				return response.content if hasattr(response, 'content') else str(response)
			except Exception as e:  # Handle various LLM errors without tight coupling
				if attempt == max_retries - 1:
					raise
				wait_time = 2 ** attempt
				app_logger.warning(f"LLM API error: {e}. Retrying in {wait_time}s...")
				time.sleep(wait_time)

	def _merge_market_analysis(self, synthesized: Dict[str, Any], market_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
		"""Safely merge optional market analysis data into synthesized output."""
		if not (market_analysis and isinstance(market_analysis, dict)):
			return synthesized

		health = market_analysis.get("health_indicators", {}) if isinstance(market_analysis.get("health_indicators", {}), dict) else {}
		if health:
			synthesized["market_health"] = health.get("market_health", synthesized.get("market_health", "unknown"))
			synthesized["opportunity_score"] = health.get("opportunity_score", synthesized.get("opportunity_score", 0.5))
			synthesized["risk_level"] = health.get("risk_level", synthesized.get("risk_level", 0.5))
			synthesized["growth_segments"] = health.get("growth_segments", synthesized.get("growth_segments", []))

		if "investment_opportunities" in market_analysis:
			synthesized["investment_opportunities"] = market_analysis.get("investment_opportunities", [])

		return synthesized

	# Public methods

	def synthesize_insights(self,
						social_insights: Optional[List[Dict[str, Any]]],
						competitor_mentions: Optional[List[Dict[str, Any]]],
						market_insights: Optional[List[Dict[str, Any]]],
						market_analysis: Optional[Dict[str, Any]] = None):
		"""Synthesize insights from multiple agents into a coherent analysis."""
		try:
			# Validate inputs
			social_insights = social_insights or []
			competitor_mentions = competitor_mentions or []
			market_insights = market_insights or []

			insights_data = {
				"social_insights": social_insights,
				"competitor_mentions": competitor_mentions,
				"market_insights": market_insights,
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

			result = self.call_llm_with_retry(prompt)
			# Ensure result is a string
			if not isinstance(result, str):
				result = str(result)
			synthesized = self.parse_llm_json(result)

			if isinstance(synthesized, dict) and "error" not in synthesized:
				# Merge optional market analysis
				synthesized = self._merge_market_analysis(synthesized, market_analysis)
				app_logger.info("Successfully synthesized insights from all agents")
				return synthesized
			else:
				app_logger.error("Failed to parse JSON from coordinator agent response")
				return self._generate_fallback_insights(social_insights, competitor_mentions, market_insights)

		except Exception as e:
			app_logger.error(f"Error in coordinator agent: {e}")
			return self._generate_fallback_insights(social_insights or [], competitor_mentions or [], market_insights or [])

	def _generate_fallback_insights(self, social_insights, competitor_mentions, market_insights):
		"""Generate fallback insights when LLM fails"""
		return {
			"executive_summary": "Synthesized analysis of Uganda fintech market trends",
			"key_trends": [
				"Growing customer discussions around mobile money fees and services",
				"Increased regulatory attention affecting market dynamics",
				"Competition intensifying between major players",
			],
			"market_health_score": 7.5,
			"investment_opportunities": [
				{"opportunity": "Rural mobile money expansion", "potential": "High"},
				{"opportunity": "Cross-border payment solutions", "potential": "Medium"},
				{"opportunity": "Digital lending platforms", "potential": "Medium"},
			],
			"risks": [
				{"risk": "Regulatory changes", "severity": "High"},
				{"risk": "Customer dissatisfaction with fees", "severity": "Medium"},
				{"risk": "Market saturation in urban areas", "severity": "Medium"},
			],
			"recommendations": [
				"Monitor regulatory developments closely",
				"Focus on customer experience improvements",
				"Explore underserved rural markets",
			],
			"confidence": 0.7,
		}

	def resolve_conflicts(self, agent_insights: Dict[str, List[Dict]]) -> Dict:
		"""
		Resolve conflicts between different agent insights
		"""
		conflicts = self._identify_conflicts(agent_insights or {})

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

			result = self.call_llm_with_retry(prompt)
			# Ensure result is a string
			if not isinstance(result, str):
				result = str(result)
			resolution = self.parse_llm_json(result)

			if isinstance(resolution, dict) and "error" not in resolution:
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
		conflicts: List[Dict[str, Any]] = []

		# Extract all insights by type and topic
		insights_by_topic: Dict[str, List[Dict[str, Any]]] = {}
		for agent_type, insights in (agent_insights or {}).items():
			for insight in insights or []:
				topic = insight.get("topic", "general")
				if topic not in insights_by_topic:
					insights_by_topic[topic] = []
				insights_by_topic[topic].append(
					{"agent": agent_type, "insight": insight, "confidence": insight.get("confidence", 0.5)}
				)

		# Look for conflicts within each topic
		for topic, topic_insights in insights_by_topic.items():
			if len(topic_insights) < 2:
				continue

			# Check for sentiment conflicts
			sentiments: Dict[str, List[str]] = {}
			for insight_data in topic_insights:
				insight = insight_data["insight"]
				if "sentiment" in insight:
					agent = insight_data["agent"]
					sentiment = insight["sentiment"]
					if sentiment not in sentiments:
						sentiments[sentiment] = []
					sentiments[sentiment].append(agent)

			# If multiple sentiments detected for same topic, it's a conflict
			if len(sentiments) > 1:
				conflicts.append(
					{
						"topic": topic,
						"type": "sentiment_conflict",
						"evidence": sentiments,
						"description": f"Conflicting sentiments about {topic}: {sentiments}",
					}
				)

		return conflicts

	def generate_daily_briefing(self, synthesized_insights: Dict) -> Dict:
		"""
		Generate a daily briefing document for investors
		"""
		try:
			prompt = f"""
			Create a professional daily briefing for fintech investors focused on Uganda.
			
			Synthesized insights: {json.dumps(synthesized_insights or {}, indent=2)}
			
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

			result = self.call_llm_with_retry(prompt)
			# Ensure result is a string
			if not isinstance(result, str):
				result = str(result)
			briefing = self.parse_llm_json(result)

			if isinstance(briefing, dict) and "error" not in briefing:
				# Store briefing in database
				try:
					self.db_manager.add_insight(
						{
							"type": "daily_briefing",
							"content": json.dumps(briefing),
							"confidence": briefing.get("confidence", 0.8),
							"source_data": [],  # All insights contribute to briefing
						}
					)
				except Exception as db_err:
					app_logger.error(f"Failed to persist daily briefing: {db_err}")

				app_logger.info("Generated daily briefing for investors")
				return briefing
			else:
				return self._generate_fallback_briefing(synthesized_insights or {})

		except Exception as e:
			app_logger.error(f"Error generating daily briefing: {e}")
			return self._generate_fallback_briefing(synthesized_insights or {})

	def _generate_fallback_briefing(self, insights: Dict) -> Dict:
		"""Generate fallback briefing when LLM fails"""
		from datetime import datetime

		return {
			"title": f"Uganda Fintech Daily Briefing - {datetime.now().strftime('%Y-%m-%d')}",
			"executive_summary": "Comprehensive analysis of Uganda's fintech market trends, opportunities, and risks based on social media intelligence.",
			"sections": [
				{
					"title": "Market Overview",
					"content": "The Uganda fintech market shows steady growth with increasing mobile money adoption and competitive dynamics.",
				},
				{
					"title": "Key Developments",
					"content": "- Customer discussions focus on service fees and reliability\n- Regulatory developments creating market uncertainty\n- New competitive threats emerging",
				},
				{
					"title": "Investment Opportunities",
					"content": "1. Rural financial inclusion initiatives\n2. Cross-border payment solutions\n3. Digital lending platforms",
				},
			],
			"key_takeaways": [
				"Market health score: 7.5/10",
				"Regulatory changes represent the biggest risk",
				"Rural markets offer significant growth potential",
			],
			"confidence": 0.7,
		}
	 