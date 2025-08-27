from .base_agent import BaseAgent
from typing import Dict, List, Any
import json
import hashlib
from datetime import datetime, timedelta

class SocialIntelAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("social_intelligence")
        self.fintech_keywords = [
            'mobile money', 'fintech', 'digital payments', 'MTN MoMo', 
            'Airtel Money', 'banking', 'loans', 'savings', 'investment',
            'chipper cash', 'payment', 'transfer', 'financial', 'wallet'
        ]
        
        # Binds the tool to LLM for dynamic queries - simplified approach
        if self.x_search_tool:
            self.llm_with_tools = self.llm  
            self.logger.info("XSearchTool available for manual use in insights generation")
        else:
            self.llm_with_tools = self.llm
            self.logger.warning("XSearchTool not available - using LLM without tools")
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data - supports both 'posts' list and 'query' string."""
        # Support query-based input for dynamic social data fetching
        if 'query' in input_data and isinstance(input_data['query'], str):
            return True
            
        # Support traditional posts input
        if 'posts' not in input_data or not isinstance(input_data['posts'], list) or not input_data['posts']:
            if hasattr(self, 'logger'):
                self.logger.error("Input data must contain either a 'query' string or a non-empty 'posts' list.")
            return False
        for post in input_data['posts']:
            if not isinstance(post, dict) or 'text' not in post or 'timestamp' not in post:
                if hasattr(self, 'logger'):
                    self.logger.error("Each post must be a dict with 'text' and 'timestamp'.")
                return False
        return True

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process social intelligence with complete pipeline: fetch, anonymize, analyze, store/cache."""
        start_time = datetime.now()
        
        if not self.validate_input(input_data):
            return {'error': 'Invalid input data'}
        
        # Create cache key for the input
        cache_key = self._create_cache_key(input_data)
        
        # Check cache first
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            self.logger.info("Returning cached social intelligence result")
            return cached_result
        
        try:
            # Step 1: Fetch social data (either from input or via XSearchTool)
            posts = self._fetch_posts(input_data)
            if not posts:
                return {'error': 'No social data available for processing'}
            
            # Step 2: Filter and process fintech-related content (already anonymized by tool)
            relevant_posts = self._filter_fintech_content(posts)
            self.logger.info(f"Filtered {len(relevant_posts)} relevant posts from {len(posts)} total")
            
            # Step 3: Store in vector database for future similarity searches
            self._store_posts_in_vector_db(relevant_posts)
            
            # Step 4: Analyze sentiment trends using LLM
            sentiment_analysis = self._analyze_sentiment_trends(relevant_posts)
            
            # Step 5: Detect emerging topics
            trending_topics = self._detect_trending_topics(relevant_posts)
            
            # Step 6: Generate insights using LLM with tool access
            insights = self._generate_social_insights_with_tools(
                sentiment_analysis, trending_topics, relevant_posts
            )
            
            # Step 7: Build final result
            result = {
                'agent': self.name,
                'timestamp': datetime.now().isoformat(),
                'query_info': input_data.get('query', f"Processed {len(posts)} posts"),
                'posts_processed': len(posts),
                'relevant_posts': len(relevant_posts),
                'sentiment_analysis': sentiment_analysis,
                'trending_topics': trending_topics,
                'insights': insights,
                'data_quality_score': self._calculate_data_quality(relevant_posts),
                'error': None
            }
            
            # Step 8: Cache the result
            self.cache_result(cache_key, result, ttl=1800)  # Cache for 30 minutes
            
            self.log_performance(start_time, result)
            return result
            
        except Exception as e:
            error_result = {
                'agent': self.name,
                'timestamp': datetime.now().isoformat(),
                'error': f"Processing failed: {str(e)}",
                'insights': []
            }
            self.logger.error(f"Social intelligence processing failed: {str(e)}")
            return error_result
    
    def _filter_fintech_content(self, posts: List[Dict]) -> List[Dict]:
        """Filter posts for fintech-related content."""
        relevant_posts = []
        
        for post in posts:
            text = post.get('text', '').lower()
            if any(keyword in text for keyword in self.fintech_keywords):
                post['relevance_score'] = self._calculate_relevance(text)
                if post['relevance_score'] > 0.5:
                    relevant_posts.append(post)
        
        return relevant_posts
    
    def _analyze_sentiment_trends(self, posts: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment trends over time."""
        prompt = f"""
        Analyze sentiment trends in these Uganda fintech discussions:

        Posts: {json.dumps([p['text'] for p in posts[:10]])}

        Return ONLY the analysis as a valid JSON object, 
        formatted exactly like below example, with no explanation, code, 
        comments or extra text. Example output:
        {{
        "overall_sentiment": "positive",
        "sentiment_score": 0.55,
        "key_sentiment_drivers": ["positive", "negative", "positive"],
        "sentiment_by_topic": {{
            "mobile_money": 0.85,
            "banking": -0.3
            }}
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
            return {"error": f"Failed to parse sentiment analysis: {response_text}"}
    
    def _detect_trending_topics(self, posts: List[Dict]) -> List[Dict]:
        """Detect trending fintech topics."""
        # Simple frequency-based trending detection
        topic_counts = {}
        current_time = datetime.now()
        
        for post in posts:
            try:
                post_time = datetime.fromisoformat(post['timestamp'])
                # Remove timezone info to make both naive for comparison
                if post_time.tzinfo is not None:
                    post_time = post_time.replace(tzinfo=None)
                
                if (current_time - post_time).days <= 7:  # Last 7 days
                    for keyword in self.fintech_keywords:
                        if keyword in post['text'].lower():
                            topic_counts[keyword] = topic_counts.get(keyword, 0) + 1
            except (ValueError, KeyError) as e:
                # Skip posts with invalid timestamps
                self.logger.warning(f"Skipping post with invalid timestamp: {e}")
                continue
        
        # Sort by frequency and return top trending
        trending = [
            {
                'topic': topic,
                'mention_count': count,
                'trend_score': count / len(posts) if posts else 0
            }
            for topic, count in sorted(
                topic_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]
        
        return trending
    
    def _generate_social_insights(self, sentiment_analysis: Dict, 
                                 trending_topics: List[Dict]) -> List[Dict]:
        """Generate actionable insights from analysis."""
        insights = []
        
        # Sentiment-based insights
        if sentiment_analysis.get('overall_sentiment') == 'negative':
            insights.append({
                'type': 'sentiment_alert',
                'severity': 'high',
                'insight': f"Negative sentiment detected in fintech discussions",
                'evidence': sentiment_analysis.get('key_sentiment_drivers', []),
                'confidence': 0.8
            })
        
        # Trending topic insights
        for topic in trending_topics[:3]:
            if topic['trend_score'] > 0.1:  # Significant trending
                insights.append({
                    'type': 'trending_topic',
                    'severity': 'medium',
                    'insight': f"{topic['topic']} showing increased discussion",
                    'evidence': f"{topic['mention_count']} mentions in last 7 days",
                    'confidence': min(topic['trend_score'] * 10, 1.0)
                })
        
        return insights
    
    def _create_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Create a cache key based on input data."""
        if 'query' in input_data:
            content = f"query:{input_data['query']}"
        else:
            # Create hash from post content for consistent caching
            post_texts = [post.get('text', '') for post in input_data.get('posts', [])]
            content = ''.join(sorted(post_texts))
        
        return f"social_intel:{hashlib.md5(content.encode()).hexdigest()}"
    
    def _fetch_posts(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch posts either from input data or via social media search."""
        if 'posts' in input_data:
            # Use provided posts
            self.logger.info(f"Using {len(input_data['posts'])} provided posts")
            return input_data['posts']
        
        elif 'query' in input_data:
            # Fetch posts using XSearchTool
            query = input_data['query']
            max_results = input_data.get('max_results', 20)
            
            # Build fintech-focused query if not already specific
            if not any(keyword in query.lower() for keyword in self.fintech_keywords):
                query = f"{query} fintech OR mobile money OR digital payments Uganda lang:en"
            
            self.logger.info(f"Fetching social data with query: {query}")
            posts = self.fetch_social_data(query, max_results)
            return posts
        
        return []
    
    def _store_posts_in_vector_db(self, posts: List[Dict[str, Any]]):
        """Store relevant posts in vector database for future similarity searches."""
        for i, post in enumerate(posts):
            try:
                doc_id = f"social_post_{datetime.now().strftime('%Y%m%d')}_{i}"
                metadata = {
                    'source': post.get('source', 'social_media'),
                    'timestamp': post.get('timestamp', datetime.now().isoformat()),
                    'relevance_score': post.get('relevance_score', 0.0),
                    'agent': self.name
                }
                self.store_in_vector_db(post['text'], metadata, doc_id)
            except Exception as e:
                self.logger.warning(f"Failed to store post {i} in vector DB: {str(e)}")
    
    def _generate_social_insights_with_tools(self, sentiment_analysis: Dict, 
                                           trending_topics: List[Dict],
                                           posts: List[Dict]) -> List[Dict]:
        """Generate insights using LLM with access to social search tools."""
        # Start with basic insights
        insights = self._generate_social_insights(sentiment_analysis, trending_topics)
        
        # Use LLM with tools for deeper analysis
        if self.llm_with_tools and len(posts) > 0:
            try:
                # Create a prompt that allows the LLM to use tools for additional context
                tool_prompt = f"""
                Based on the social media analysis of {len(posts)} posts about fintech in Uganda:
                
                Sentiment Analysis: {json.dumps(sentiment_analysis)}
                Trending Topics: {json.dumps(trending_topics)}
                
                Generate 2-3 additional actionable insights for fintech companies in Uganda. 
                You can use the x_search tool to gather more context if needed.
                
                Return insights as JSON array with format:
                [{{"type": "market_insight", "severity": "medium", "insight": "description", "evidence": "supporting_data", "confidence": 0.8}}]
                """
                
                response = self.llm_with_tools.invoke(tool_prompt)
                
                # Extract and parse LLM response
                if hasattr(response, "content"):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Try to extract JSON from response
                try:
                    additional_insights = json.loads(response_text)
                    if isinstance(additional_insights, list):
                        insights.extend(additional_insights)
                        self.logger.info(f"Added {len(additional_insights)} tool-enhanced insights")
                except json.JSONDecodeError:
                    self.logger.warning("Could not parse additional insights from LLM response")
                    
            except Exception as e:
                self.logger.error(f"Failed to generate tool-enhanced insights: {str(e)}")
        
        return insights
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score for fintech content."""
        score = 0.0
        text_lower = text.lower()
        
        # Keyword matching with weights
        keyword_weights = {
            'mobile money': 0.3,
            'fintech': 0.2,
            'digital payments': 0.2,
            'banking': 0.15,
            'mtn momo': 0.25,
            'airtel money': 0.25,
            'chipper cash': 0.25,
            'payment': 0.1,
            'transfer': 0.1,
            'financial': 0.1,
            'uganda': 0.15
        }
        
        for keyword, weight in keyword_weights.items():
            if keyword in text_lower:
                score += weight
        
        return min(score, 1.0)
    
    def _calculate_data_quality(self, posts: List[Dict]) -> float:
        """Calculate overall data quality score."""
        if not posts:
            return 0.0
        
        quality_factors = []
        
        # Text length quality (not too short, not too long)
        avg_length = sum(len(post['text']) for post in posts) / len(posts)
        length_score = 1.0 if 50 <= avg_length <= 500 else 0.5
        quality_factors.append(length_score)
        
        # Language quality (English content)
        english_posts = sum(1 for post in posts if self._is_english(post['text']))
        language_score = english_posts / len(posts)
        quality_factors.append(language_score)
        
        # Relevance quality
        avg_relevance = sum(post.get('relevance_score', 0) for post in posts) / len(posts)
        quality_factors.append(avg_relevance)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _is_english(self, text: str) -> bool:
        """Simple English language detection."""
        english_indicators = ['the', 'and', 'is', 'to', 'of', 'in', 'for', 'on', 'with']
        text_lower = text.lower()
        matches = sum(1 for word in english_indicators if word in text_lower)
        return matches >= 3