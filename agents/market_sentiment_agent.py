from .base_agent import BaseAgent
from typing import Dict, List, Any
from datetime import datetime, timedelta
import json
import hashlib
import re
from database.db_manager import DatabaseManager

class MarketSentimentAgent(BaseAgent):
    """Agent for analyzing overall market sentiment and trends in Uganda's fintech ecosystem."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the MarketSentimentAgent with base infrastructure."""
        super().__init__("market_sentiment")
        
        # Market sentiment specific configuration
        self.config = config or {}
        
        # Initialize database connection for historical data
        self.db_manager = DatabaseManager()
        
        # Define market segments to track
        self.market_segments = self.config.get("market_segments", [
            "mobile_money", "digital_banking", "lending", "savings", 
            "investments", "cross_border", "payments", "rural_finance"
        ])
        
        # Define risk factors to monitor
        self.risk_factors = self.config.get("risk_factors", [
            "regulatory", "competition", "technology", "economic", 
            "security", "adoption", "infrastructure"
        ])
        
        # Opportunity indicators
        self.opportunity_indicators = [
            "growth", "unmet need", "innovation", "gap", "potential",
            "underserved", "expansion", "demand", "emerging"
        ]
        
        self.logger.info(f"MarketSentimentAgent initialized with {len(self.market_segments)} market segments")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for market sentiment analysis."""
        if not isinstance(input_data, dict):
            self.logger.error("Input must be a dictionary")
            return False
        
        # Support different input types
        has_time_period = "days" in input_data or "hours" in input_data
        has_posts = "posts" in input_data and isinstance(input_data["posts"], list)
        has_segment = "segment" in input_data and input_data["segment"] in self.market_segments
        
        if not (has_time_period or has_posts or has_segment):
            self.logger.error("Input must contain 'days', 'hours', 'posts', or valid 'segment'")
            return False
        
        return True
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market sentiment data and generate insights."""
        start_time = datetime.now()
        
        if not self.validate_input(input_data):
            return {"error": "Invalid input data"}
        
        # Create cache key for the input
        cache_key = self._create_cache_key(input_data)
        
        # Check cache first
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            self.logger.info("Returning cached market sentiment result")
            return cached_result
        
        try:
            # Step 1: Gather relevant data
            posts = self._gather_market_data(input_data)
            if not posts:
                return {"error": "No market data available for analysis"}
            
            # Step 2: Analyze overall market sentiment
            market_sentiment = self._analyze_market_sentiment(posts)
            
            # Step 3: Identify trends by segment
            segment_trends = self._analyze_segment_trends(posts)
            
            # Step 4: Detect investment opportunities
            opportunities = self._detect_investment_opportunities(posts, segment_trends)
            
            # Step 5: Assess market risks
            risks = self._assess_market_risks(posts)
            
            # Step 6: Generate market health indicators
            health_indicators = self._generate_market_health_indicators(
                market_sentiment, segment_trends, risks
            )
            
            # Step 7: Build final result
            result = {
                "agent": self.name,
                "timestamp": datetime.now().isoformat(),
                "time_period": self._get_time_period_description(input_data),
                "posts_analyzed": len(posts),
                "market_sentiment": market_sentiment,
                "segment_trends": segment_trends,
                "investment_opportunities": opportunities,
                "market_risks": risks,
                "health_indicators": health_indicators,
                "error": None
            }
            
            # Step 8: Cache the result
            self.cache_result(cache_key, result, ttl=3600)  # Cache for 1 hour
            
            self.log_performance(start_time, result)
            return result
            
        except Exception as e:
            error_result = {
                "agent": self.name,
                "timestamp": datetime.now().isoformat(),
                "error": f"Processing failed: {str(e)}",
                "health_indicators": {"market_health": "unknown"}
            }
            self.logger.error(f"Market sentiment processing failed: {str(e)}")
            return error_result
    
    def _create_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Create a cache key based on input data."""
        if "days" in input_data:
            content = f"days:{input_data['days']}"
        elif "hours" in input_data:
            content = f"hours:{input_data['hours']}"
        elif "segment" in input_data:
            content = f"segment:{input_data['segment']}"
        else:
            # Create hash from post content for consistent caching
            post_texts = [post.get('text', '') for post in input_data.get('posts', [])]
            content = ''.join(sorted(post_texts))
        
        return f"market_sentiment:{hashlib.md5(content.encode()).hexdigest()}"
    
    def _gather_market_data(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather relevant market data based on input parameters."""
        posts = []
        
        # If posts are directly provided
        if "posts" in input_data:
            posts = input_data["posts"]
            self.logger.info(f"Using {len(posts)} provided posts for market analysis")
        
        # If time period is specified
        elif "days" in input_data or "hours" in input_data:
            days = input_data.get("days", 0)
            hours = input_data.get("hours", 0)
            total_hours = days * 24 + hours
            
            # Get posts from database
            db_posts = self.db_manager.get_recent_posts(hours=total_hours)
            
            # Convert to the format we need
            for post in db_posts:
                posts.append({
                    "text": post.cleaned_content or post.content,
                    "source": post.source,
                    "timestamp": post.timestamp.isoformat() if post.timestamp else datetime.now().isoformat(),
                    "sentiment": post.sentiment,
                    "sentiment_score": post.sentiment_score,
                    "topics": json.loads(post.topics) if post.topics else []
                })
            
            self.logger.info(f"Retrieved {len(posts)} posts from the last {total_hours} hours")
        
        # If specific segment is requested
        elif "segment" in input_data:
            segment = input_data["segment"]
            
            # Get posts related to this segment from vector DB
            segment_posts = self.query_vector_db(segment, k=50)
            
            # Convert to the format we need
            for result in segment_posts:
                posts.append({
                    "text": result["text"],
                    "source": result["metadata"].get("source", "unknown"),
                    "timestamp": result["metadata"].get("timestamp", datetime.now().isoformat()),
                    "sentiment": result["metadata"].get("sentiment", "neutral"),
                    "sentiment_score": result["metadata"].get("sentiment_score", 0.5),
                    "topics": result["metadata"].get("topics", [])
                })
            
            self.logger.info(f"Retrieved {len(posts)} posts related to {segment}")
        
        return posts
    
    def _analyze_market_sentiment(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall market sentiment from collected posts."""
        # If posts already have sentiment, use that
        if all(post.get("sentiment") for post in posts):
            positive_count = sum(1 for post in posts if post.get("sentiment") == "positive")
            negative_count = sum(1 for post in posts if post.get("sentiment") == "negative")
            neutral_count = sum(1 for post in posts if post.get("sentiment") == "neutral")
            
            total = len(posts)
            if total == 0:
                return {"overall_sentiment": "neutral", "sentiment_score": 0.5}
            
            positive_ratio = positive_count / total
            negative_ratio = negative_count / total
            
            # Calculate overall sentiment score (0-1 scale)
            sentiment_score = 0.5 + (positive_ratio - negative_ratio) / 2
            
            # Determine overall sentiment
            if sentiment_score > 0.6:
                overall_sentiment = "positive"
            elif sentiment_score < 0.4:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            return {
                "overall_sentiment": overall_sentiment,
                "sentiment_score": sentiment_score,
                "distribution": {
                    "positive": positive_ratio,
                    "negative": negative_ratio,
                    "neutral": neutral_count / total
                }
            }
        
        # If posts don't have sentiment, use LLM to analyze
        else:
            prompt = f"""
            Analyze the overall market sentiment in these Uganda fintech discussions:

            Posts: {json.dumps([p['text'] for p in posts[:15]])}

            Return ONLY the analysis as a valid JSON object, 
            formatted exactly like below example, with no explanation, code, 
            comments or extra text. Example output:
            {{
            "overall_sentiment": "positive",
            "sentiment_score": 0.65,
            "distribution": {{
                "positive": 0.55,
                "negative": 0.25,
                "neutral": 0.20
            }},
            "key_drivers": ["mobile money growth", "improved access", "regulatory concerns"]
            }}
            """
            
            response = self.llm.invoke(prompt)
            
            # Extract text from AIMessage or use str fallback
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse sentiment analysis: {response_text}")
                return {
                    "overall_sentiment": "neutral",
                    "sentiment_score": 0.5,
                    "distribution": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                    "error": "Failed to parse sentiment analysis"
                }
    
    def _analyze_segment_trends(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze trends by market segment."""
        segment_mentions = {segment: 0 for segment in self.market_segments}
        segment_sentiment = {segment: {"positive": 0, "negative": 0, "neutral": 0} for segment in self.market_segments}
        
        # Count mentions and sentiment by segment
        for post in posts:
            text = post.get("text", "").lower()
            sentiment = post.get("sentiment", "neutral")
            
            for segment in self.market_segments:
                # Check if segment is mentioned (using segment name with underscores replaced by spaces)
                segment_term = segment.replace("_", " ")
                if segment_term in text:
                    segment_mentions[segment] += 1
                    segment_sentiment[segment][sentiment] += 1
        
        # Calculate trends and momentum
        trends = []
        for segment, mentions in segment_mentions.items():
            if mentions > 0:
                # Calculate sentiment distribution
                total = sum(segment_sentiment[segment].values())
                positive_ratio = segment_sentiment[segment]["positive"] / total if total > 0 else 0
                negative_ratio = segment_sentiment[segment]["negative"] / total if total > 0 else 0
                
                # Calculate sentiment score
                sentiment_score = 0.5 + (positive_ratio - negative_ratio) / 2
                
                # Calculate mention frequency (as percentage of total posts)
                mention_frequency = mentions / len(posts)
                
                # Determine trend direction
                if sentiment_score > 0.6 and mention_frequency > 0.1:
                    trend_direction = "rising"
                elif sentiment_score < 0.4 and mention_frequency > 0.1:
                    trend_direction = "declining"
                else:
                    trend_direction = "stable"
                
                trends.append({
                    "segment": segment,
                    "mentions": mentions,
                    "mention_frequency": mention_frequency,
                    "sentiment_score": sentiment_score,
                    "trend_direction": trend_direction,
                    "momentum": mention_frequency * sentiment_score  # Combined metric
                })
        
        # Sort by momentum (highest first)
        return sorted(trends, key=lambda x: x["momentum"], reverse=True)
    
    def _detect_investment_opportunities(self, posts: List[Dict[str, Any]], segment_trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect potential investment opportunities based on post content and segment trends."""
        # Extract top segments by momentum
        top_segments = [t["segment"] for t in segment_trends[:3]]
        
        # Use LLM to identify opportunities
        post_texts = [p["text"] for p in posts[:20]]  # Use a subset of posts
        
        prompt = f"""
        Analyze these Uganda fintech discussions to identify 3-5 specific investment opportunities:

        Posts: {json.dumps(post_texts)}

        Top trending segments: {", ".join(top_segments)}

        Return ONLY a JSON array of investment opportunities, with each opportunity having:
        1. "segment" - The market segment (e.g., mobile_money, lending)
        2. "opportunity" - Brief description of the opportunity
        3. "evidence" - Evidence from the posts
        4. "confidence" - Confidence score (0-1)

        Format exactly like this example, with no explanation, code, comments or extra text:
        [
            {{
                "segment": "mobile_money",
                "opportunity": "Rural agent expansion for mobile money services",
                "evidence": "Multiple mentions of lack of agents in rural areas",
                "confidence": 0.85
            }},
            {{
                "segment": "lending",
                "opportunity": "Digital micro-loans for small businesses",
                "evidence": "Complaints about lack of access to small business capital",
                "confidence": 0.78
            }}
        ]
        """
        
        response = self.llm.invoke(prompt)
        
        # Extract text from AIMessage or use str fallback
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)
        
        try:
            opportunities = json.loads(response_text)
            return opportunities
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse investment opportunities: {response_text}")
            return []
    
    def _assess_market_risks(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assess market risks based on post content."""
        # Count risk factor mentions
        risk_mentions = {risk: 0 for risk in self.risk_factors}
        
        for post in posts:
            text = post.get("text", "").lower()
            for risk in self.risk_factors:
                if risk in text:
                    risk_mentions[risk] += 1
        
        # Use LLM to assess risks from the most mentioned factors
        top_risks = sorted(risk_mentions.items(), key=lambda x: x[1], reverse=True)[:3]
        top_risk_factors = [risk for risk, count in top_risks if count > 0]
        
        if not top_risk_factors:
            return []
        
        # Filter posts that mention top risks
        risk_posts = []
        for post in posts:
            text = post.get("text", "").lower()
            if any(risk in text for risk, _ in top_risks):
                risk_posts.append(post["text"])
        
        if not risk_posts:
            return []
        
        prompt = f"""
        Analyze these Uganda fintech discussions to assess market risks:

        Posts related to risks: {json.dumps(risk_posts[:15])}

        Top risk factors mentioned: {", ".join(top_risk_factors)}

        Return ONLY a JSON array of risk assessments, with each risk having:
        1. "risk_factor" - The type of risk (e.g., regulatory, competition)
        2. "description" - Brief description of the specific risk
        3. "severity" - Risk severity (low, medium, high)
        4. "evidence" - Evidence from the posts
        5. "mitigation" - Potential mitigation strategy

        Format exactly like this example, with no explanation, code, comments or extra text:
        [
            {{
                "risk_factor": "regulatory",
                "description": "New mobile money taxation policy",
                "severity": "high",
                "evidence": "Multiple discussions about government introducing new taxes",
                "mitigation": "Diversify revenue streams beyond transaction fees"
            }},
            {{
                "risk_factor": "competition",
                "description": "New market entrant with lower fees",
                "severity": "medium",
                "evidence": "Mentions of new competitor with aggressive pricing",
                "mitigation": "Focus on service quality and reliability"
            }}
        ]
        """
        
        response = self.llm.invoke(prompt)
        
        # Extract text from AIMessage or use str fallback
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)
        
        try:
            risks = json.loads(response_text)
            return risks
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse market risks: {response_text}")
            return []
    
    def _generate_market_health_indicators(self, 
                                         market_sentiment: Dict[str, Any],
                                         segment_trends: List[Dict[str, Any]],
                                         risks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate market health indicators based on sentiment, trends, and risks."""
        # Calculate overall market health score (0-1 scale)
        sentiment_score = market_sentiment.get("sentiment_score", 0.5)
        
        # Calculate average segment momentum
        avg_momentum = sum(s.get("momentum", 0) for s in segment_trends) / len(segment_trends) if segment_trends else 0.5
        
        # Calculate risk severity
        risk_severity = 0
        for risk in risks:
            severity = {"low": 0.25, "medium": 0.5, "high": 1.0}.get(risk.get("severity", "medium"), 0.5)
            risk_severity += severity
        
        # Normalize risk severity (0-1 scale, higher means more risk)
        normalized_risk = min(1.0, risk_severity / 3) if risks else 0.3
        
        # Calculate overall health score (higher is better)
        health_score = (sentiment_score * 0.5) + (avg_momentum * 0.3) - (normalized_risk * 0.2)
        health_score = max(0, min(1, health_score))  # Ensure it's between 0 and 1
        
        # Determine market health status
        if health_score >= 0.7:
            market_health = "strong"
        elif health_score >= 0.5:
            market_health = "stable"
        elif health_score >= 0.3:
            market_health = "caution"
        else:
            market_health = "weak"
        
        # Determine opportunity score
        opportunity_score = (sentiment_score * 0.4) + (avg_momentum * 0.4) + ((1 - normalized_risk) * 0.2)
        opportunity_score = max(0, min(1, opportunity_score))  # Ensure it's between 0 and 1
        
        # Generate top growth segments
        growth_segments = [s["segment"] for s in segment_trends if s.get("trend_direction") == "rising"][:3]
        
        return {
            "market_health": market_health,
            "health_score": health_score,
            "opportunity_score": opportunity_score,
            "risk_level": normalized_risk,
            "growth_segments": growth_segments,
            "sentiment_indicator": market_sentiment.get("overall_sentiment", "neutral"),
            "momentum_indicator": "positive" if avg_momentum > 0.5 else "neutral" if avg_momentum > 0.3 else "negative"
        }
    
    def _get_time_period_description(self, input_data: Dict[str, Any]) -> str:
        """Get a description of the time period covered by the analysis."""
        if "days" in input_data:
            return f"Last {input_data['days']} days"
        elif "hours" in input_data:
            return f"Last {input_data['hours']} hours"
        elif "segment" in input_data:
            return f"Segment analysis: {input_data['segment']}"
        else:
            return "Custom data set"
