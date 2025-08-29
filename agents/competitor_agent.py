from .base_agent import BaseAgent
from typing import Dict, List, Any
from datetime import datetime, timedelta
import json
import re

class CompetitorAnalysisAgent(BaseAgent):
    """Agent for analyzing competitor mentions and sentiment in social media data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the CompetitorAnalysisAgent with base infrastructure."""
        super().__init__("competitor_analysis")
        
        # Competitor-specific configuration
        self.config = config or {}
        self.competitors = self.config.get("competitors", [
            "MTN MoMo", "Airtel Money", "Chipper Cash", "Stanbic Bank", 
            "Centenary Bank", "Equity Bank", "DFCU Bank"
        ])
        
        self.logger.info(f"CompetitorAnalysisAgent initialized with {len(self.competitors)} competitors")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for competitor analysis."""
        if not isinstance(input_data, dict):
            self.logger.error("Input must be a dictionary")
            return False
        
        # Support both query-based and posts-based analysis
        has_query = "query" in input_data or "competitor" in input_data
        has_posts = "posts" in input_data and isinstance(input_data["posts"], list)
        has_hours = "hours" in input_data
        
        if not (has_query or has_posts or has_hours):
            self.logger.error("Input must contain 'query', 'competitor', 'posts', or 'hours'")
            return False
        
        return True
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method for competitor analysis."""
        if not self.validate_input(input_data):
            return {"error": "Invalid input data"}
        
        # Create cache key
        cache_key = self._create_cache_key(input_data)
        
        # Check cache first
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            self.logger.info("Returning cached competitor analysis")
            return cached_result
        
        try:
            # Determine processing type
            if "posts" in input_data:
                # Analyze provided posts
                result = self._analyze_posts(input_data["posts"])
            elif "competitor" in input_data:
                # Analyze specific competitor
                result = self._analyze_competitor(input_data["competitor"], 
                                                input_data.get("hours", 24))
            else:
                # General competitive intelligence
                result = self._generate_competitive_intelligence(
                    input_data.get("hours", 24)
                )
            
            # Cache the result
            self.cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in competitor analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _create_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Create cache key for competitor analysis."""
        key_parts = ["competitor_analysis"]
        
        if "competitor" in input_data:
            key_parts.append(f"comp_{input_data['competitor']}")
        if "hours" in input_data:
            key_parts.append(f"hours_{input_data['hours']}")
        if "posts" in input_data:
            key_parts.append(f"posts_{len(input_data['posts'])}")
            
        return "_".join(key_parts)
    
    def _analyze_posts(self, posts: List[Dict]) -> Dict[str, Any]:
        """Analyze provided posts for competitor mentions."""
        competitor_mentions = []
        
        for i, post in enumerate(posts):
            insights = self._extract_competitor_insights_from_post(i, post.get('text', ''))
            competitor_mentions.extend(insights)
        
        # Generate summary
        summary = self._generate_summary_insights(competitor_mentions)
        
        return {
            "analysis_type": "posts",
            "total_posts_analyzed": len(posts),
            "competitor_mentions": competitor_mentions,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_competitor(self, competitor: str, hours: int = 24) -> Dict[str, Any]:
        """Analyze mentions of a specific competitor."""
        # Search vector database for competitor mentions
        query_results = self.query_vector_db(competitor, k=50)
        
        competitor_mentions = []
        for result in query_results:
            # Filter by timestamp if available
            post_time = result.get("metadata", {}).get("timestamp")
            if self._is_recent_post(post_time, hours):
                analysis = self._analyze_competitor_sentiment(
                    result.get("text", ""), competitor
                )
                
                mention = {
                    "post_id": result.get("metadata", {}).get("post_id", f"doc_{len(competitor_mentions)}"),
                    "competitor": competitor,
                    "sentiment": analysis.get("sentiment", "neutral"),
                    "context": result.get("text", "")[:500],
                    "extracted_insights": analysis.get("key_points", []),
                    "confidence": analysis.get("confidence", 0.5),
                    "timestamp": post_time
                }
                
                competitor_mentions.append(mention)
        
        # Store results in vector database
        if competitor_mentions:
            self._store_competitor_analysis(competitor_mentions)
        
        return {
            "analysis_type": "competitor_specific",
            "competitor": competitor,
            "time_period_hours": hours,
            "total_mentions": len(competitor_mentions),
            "mentions": competitor_mentions,
            "summary": self._generate_summary_insights(competitor_mentions),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_competitive_intelligence(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive competitive intelligence report."""
        all_mentions = []
        competitor_summaries = {}
        
        for competitor in self.competitors:
            # Use the single competitor analysis method
            comp_analysis = self._analyze_competitor(competitor, hours)
            competitor_summaries[competitor] = comp_analysis
            all_mentions.extend(comp_analysis.get("mentions", []))
        
        # Generate overall insights
        overall_summary = self._generate_summary_insights(all_mentions)
        
        return {
            "analysis_type": "comprehensive",
            "time_period_hours": hours,
            "competitors_analyzed": len(self.competitors),
            "total_mentions": len(all_mentions),
            "competitor_summaries": competitor_summaries,
            "overall_summary": overall_summary,
            "competitive_landscape": self._generate_competitive_landscape(competitor_summaries),
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_competitor_insights_from_post(self, post_id: int, content: str) -> List[Dict]:
        """Extract competitor insights from a single post."""
        insights = []
        
        for competitor in self.competitors:
            if competitor.lower() in content.lower():
                analysis = self._analyze_competitor_sentiment(content, competitor)
                
                insight = {
                    "post_id": post_id,
                    "competitor": competitor,
                    "sentiment": analysis.get("sentiment", "neutral"),
                    "context": content[:500],
                    "extracted_insights": analysis.get("key_points", []),
                    "confidence": analysis.get("confidence", 0.5),
                    "timestamp": datetime.now().isoformat()
                }
                
                insights.append(insight)
        
        return insights
    
    def _analyze_competitor_sentiment(self, content: str, competitor: str) -> Dict:
        """Analyze sentiment for a specific competitor mention using LLM."""
        try:
            prompt = f"""
            Analyze this social media content mentioning {competitor} in Uganda's fintech market:
            
            Content: {content}
            
            Provide analysis in JSON format:
            {{
                "sentiment": "positive/negative/neutral",
                "key_points": ["insight1", "insight2"],
                "confidence": 0.8,
                "competitive_aspect": "pricing/features/service/brand"
            }}
            
            Focus on: customer sentiment, product features, pricing, service quality, competitive positioning.
            """
            
            response = self.llm.invoke(prompt)
            
            # Extract JSON from response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_analysis(competitor)
                
        except Exception as e:
            self.logger.error(f"Error analyzing competitor sentiment: {e}")
            return self._fallback_analysis(competitor)
    
    def _fallback_analysis(self, competitor: str) -> Dict:
        """Fallback analysis when LLM analysis fails."""
        return {
            "sentiment": "neutral",
            "key_points": [f"Mention of {competitor} detected"],
            "confidence": 0.3,
            "competitive_aspect": "general"
        }
    
    def _generate_summary_insights(self, analyzed_mentions: List[Dict]) -> List[Dict]:
        """Generate summary insights from analyzed competitor mentions."""
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
                neutral_pct = (data["sentiments"]["neutral"] / total_mentions) * 100
                
                # Determine overall sentiment
                if positive_pct > negative_pct and positive_pct > neutral_pct:
                    overall_sentiment = "positive"
                    confidence = positive_pct / 100
                elif negative_pct > positive_pct and negative_pct > neutral_pct:
                    overall_sentiment = "negative"
                    confidence = negative_pct / 100
                else:
                    overall_sentiment = "neutral"
                    confidence = max(neutral_pct, 50) / 100
                
                summary_insights.append({
                    "competitor": competitor,
                    "overall_sentiment": overall_sentiment,
                    "confidence": confidence,
                    "total_mentions": total_mentions,
                    "sentiment_breakdown": {
                        "positive": f"{positive_pct:.1f}%",
                        "negative": f"{negative_pct:.1f}%",
                        "neutral": f"{neutral_pct:.1f}%"
                    },
                    "insight": f"{competitor}: {overall_sentiment} sentiment with {total_mentions} mentions"
                })
            
            return summary_insights
            
        except Exception as e:
            self.logger.error(f"Error generating summary insights: {e}")
            return []
    
    def _generate_competitive_landscape(self, competitor_summaries: Dict) -> Dict:
        """Generate competitive landscape analysis."""
        landscape = {
            "market_leaders": [],
            "sentiment_leaders": [],
            "mention_volume": {},
            "key_insights": []
        }
        
        try:
            # Analyze mention volume and sentiment
            for competitor, summary in competitor_summaries.items():
                mentions = summary.get("total_mentions", 0)
                landscape["mention_volume"][competitor] = mentions
                
                # Get overall sentiment from summary
                comp_summary = summary.get("summary", [])
                if comp_summary:
                    sentiment_data = comp_summary[0]
                    if sentiment_data.get("overall_sentiment") == "positive":
                        landscape["sentiment_leaders"].append({
                            "competitor": competitor,
                            "confidence": sentiment_data.get("confidence", 0)
                        })
            
            # Sort by mention volume
            sorted_mentions = sorted(landscape["mention_volume"].items(), 
                                   key=lambda x: x[1], reverse=True)
            landscape["market_leaders"] = [comp for comp, _ in sorted_mentions[:3]]
            
            # Sort sentiment leaders by confidence
            landscape["sentiment_leaders"].sort(key=lambda x: x["confidence"], reverse=True)
            
            # Generate key insights
            if sorted_mentions:
                top_competitor = sorted_mentions[0]
                landscape["key_insights"].append(
                    f"{top_competitor[0]} has highest mention volume with {top_competitor[1]} mentions"
                )
            
            if landscape["sentiment_leaders"]:
                top_sentiment = landscape["sentiment_leaders"][0]
                landscape["key_insights"].append(
                    f"{top_sentiment['competitor']} leads in positive sentiment"
                )
            
        except Exception as e:
            self.logger.error(f"Error generating competitive landscape: {e}")
        
        return landscape
    
    def _store_competitor_analysis(self, competitor_mentions: List[Dict]):
        """Store competitor analysis results in vector database."""
        for mention in competitor_mentions:
            try:
                doc_id = f"competitor_{mention['competitor']}_{mention['post_id']}_{datetime.now().strftime('%Y%m%d%H%M')}"
                
                metadata = {
                    "type": "competitor_analysis",
                    "competitor": mention["competitor"],
                    "sentiment": mention["sentiment"],
                    "confidence": mention["confidence"],
                    "timestamp": mention.get("timestamp", datetime.now().isoformat()),
                    "agent": self.name
                }
                
                content = f"Competitor: {mention['competitor']}\nSentiment: {mention['sentiment']}\nInsights: {mention['extracted_insights']}\nContext: {mention['context']}"
                
                self.store_in_vector_db(content, metadata, doc_id)
                
            except Exception as e:
                self.logger.warning(f"Failed to store competitor mention: {str(e)}")
    
    def _is_recent_post(self, timestamp_str: str, hours: int) -> bool:
        """Check if a post is within the specified time window."""
        if not timestamp_str:
            return True  # Include posts without timestamps
        
        try:
            post_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            current_time = datetime.now()
            time_diff = current_time - post_time.replace(tzinfo=None)
            return time_diff <= timedelta(hours=hours)
        except Exception:
            return True  # Include posts with invalid timestamps