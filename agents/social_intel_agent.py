from .base_agent import BaseAgent
from config import Config
from typing import Dict, List, Any
import json
import hashlib
import re  
from datetime import datetime, timedelta

class SocialIntelAgent(BaseAgent):
    def __init__(self, config=None):
        super().__init__(name="social_intelligence", agent_type="social_intelligence", config=config)
        
        # Use fintech keywords from config
        self.fintech_keywords = Config.get_all_fintech_keywords()
        
        # Configuration from config.py - Use the new get_agent_config method
        self.cache_ttl = Config.get_agent_config("social_intelligence", "cache_ttl", Config.DEFAULT_CACHE_TTL)
        self.max_posts = Config.get_agent_config("social_intelligence", "max_posts", Config.DEFAULT_MAX_POSTS)
        self.relevance_threshold = Config.get_agent_config("social_intelligence", "relevance_threshold", Config.DEFAULT_RELEVANCE_THRESHOLD)
        self.trending_days = Config.get_agent_config("social_intelligence", "trending_days", 7)
        self.trending_threshold = Config.get_agent_config("social_intelligence", "trending_threshold", 0.1)
        
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
            # Step 1: Fetchs social data (either from input or via XSearchTool)
            posts = self._fetch_posts(input_data)
            if not posts:
                return {'error': 'No social data available for processing'}
            
            # Step 2: Filters and processes fintech-related content (already anonymized by tool)
            relevant_posts = self._filter_fintech_content(posts)
            self.logger.info(f"Filtered {len(relevant_posts)} relevant posts from {len(posts)} total")
            
            # Step 3: Stores filtered posts in vector database for future similarity searches
            self._store_posts_in_vector_db(relevant_posts)
            
            # Step 4: Analyzes sentiment trends using LLM
            sentiment_analysis = self._analyze_sentiment_trends(relevant_posts)
            
            # Step 5: Detects emerging topics
            trending_topics = self._detect_trending_topics(relevant_posts)
            
            # Step 6: Generates insights using LLM with tool access
            insights = self._generate_social_insights_with_tools(
                sentiment_analysis, trending_topics, relevant_posts
            )
            
            # Step 7: Builds final result
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
                if post['relevance_score'] > 0.45: 
                    relevant_posts.append(post)
        
        return relevant_posts
    
    def _analyze_sentiment_trends(self, posts: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment trends over time with robust JSON parsing."""
        prompt = f"""
        Analyze sentiment trends in these Uganda fintech discussions:

        Posts: {json.dumps([p['text'] for p in posts[:10]], ensure_ascii=False)}

        Return ONLY a valid JSON object with no additional text, explanations, or markdown formatting.
        Use this exact structure:

        {{
            "overall_sentiment": "positive/negative/neutral",
            "sentiment_score": 0.55,
            "key_sentiment_drivers": ["driver1", "driver2", "driver3"],
            "sentiment_by_topic": {{
                "mobile_money": 0.85,
                "banking": -0.3
            }}
        }}
        """
        
        response_text = ""  # Initialize variable to avoid scope issues
        
        try:
            response = self.llm.invoke(prompt)
            
            # Extract text from response
            if hasattr(response, "content") and response.content:
                content = response.content
                if isinstance(content, str):
                    response_text = content.strip()
                elif isinstance(content, (list, tuple)):
                    response_text = ' '.join(str(item) for item in content).strip()
                else:
                    response_text = str(content).strip()
            else:
                response_text = str(response).strip() if response else ""
            
            # Clean the response - remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
            
            # Parse JSON
            result = json.loads(response_text)
            self.logger.info("Successfully parsed sentiment analysis JSON")
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            self.logger.error(f"Raw response: {response_text[:200]}...")
            
            # Return fallback structure
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.5,
                "key_sentiment_drivers": ["insufficient_data"],
                "sentiment_by_topic": {},
                "error": f"JSON parsing failed: {str(e)}"
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.5,
                "key_sentiment_drivers": ["analysis_error"],
                "sentiment_by_topic": {},
                "error": f"Analysis failed: {str(e)}"
            }
    
    def _detect_trending_topics(self, posts: List[Dict]) -> List[Dict]:
        """Detect trending fintech topics."""
        topic_counts = {}
        current_time = datetime.now()
        
        for post in posts:
            try:
                post_time = datetime.fromisoformat(post['timestamp'])
                # Remove timezone info to make both naive for comparison
                if post_time.tzinfo is not None:
                    post_time = post_time.replace(tzinfo=None)
                
                # Use config value for trending days
                trending_days = self.trending_days if isinstance(self.trending_days,int) else 7
                if (current_time - post_time).days <= trending_days:
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
            max_results = input_data.get('max_results', self.max_posts)  # Use config default
            
            # Ensure max_results is an integer
            if not isinstance(max_results, int):
                max_results = self.max_posts or 10
            
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
        """Generate insights using LLM with tool context."""
        insights = []
        
        try:
            if self.llm_with_tools and len(posts) > 0:
                # Create a comprehensive prompt for better structured output
                tool_prompt = f"""
Based on the social media analysis for fintech content, provide actionable insights.

SENTIMENT DATA:
- Overall sentiment: {sentiment_analysis.get('overall_sentiment', 'neutral')}
- Sentiment score: {sentiment_analysis.get('sentiment_score', 0.5)}

TRENDING TOPICS:
{chr(10).join([f"- {topic['topic']}: {topic['mention_count']} mentions" for topic in trending_topics[:5]])}

SAMPLE POSTS:
{chr(10).join([f"- {post['text'][:100]}..." for post in posts[:3]])}

Please provide 3-5 specific, actionable insights in this exact format:
INSIGHT: [Your insight here]
INSIGHT: [Another insight here]
INSIGHT: [Third insight here]

Focus on:
1. Market opportunities in Uganda's fintech sector
2. Customer sentiment patterns
3. Competitive landscape observations
4. Regulatory or technology trends
5. Risk factors or challenges
"""

                response = self.llm_with_tools.invoke(tool_prompt)
                
                # Improved parsing logic
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Parse insights using multiple patterns
                insight_patterns = [
                    r'INSIGHT:\s*(.+)',           # INSIGHT: format
                    r'^\d+\.\s*(.+)',            # 1. numbered format  
                    r'^-\s*(.+)',                # - bullet format
                    r'^•\s*(.+)',                # • bullet format
                    r'^\*\s*(.+)'                # * bullet format
                ]
                
                found_insights = []
                
                for pattern in insight_patterns:
                    if response_text and isinstance(response_text,str):
                        matches = re.findall(pattern, response_text, re.MULTILINE)
                        found_insights.extend(matches)
                
                # If structured parsing fails, extract sentences as insights
                if not found_insights:
                    # Fix for line 384 - add safety checks
                    if response_text and isinstance(response_text, str):
                        sentences = response_text.split('.')
                        found_insights = [s.strip() for s in sentences 
                                        if s.strip() and len(s.strip()) > 20 and len(s.strip()) < 200]
                
                # Convert to structured format
                for i, insight_text in enumerate(found_insights[:5]):  # Limit to 5 insights
                    if insight_text and len(insight_text.strip()) > 10:
                        insights.append({
                            'type': 'llm_generated',
                            'insight': insight_text.strip(),
                            'confidence': 0.8 - (i * 0.1),  # Decreasing confidence
                            'timestamp': datetime.now().isoformat(),
                            'source': 'tool_enhanced_llm'
                        })
                
                if insights:
                    self.logger.info(f"Generated {len(insights)} tool-enhanced insights")
                else:
                    self.logger.warning("LLM response could not be parsed into structured insights")
                    # Add a fallback insight
                    insights.append({
                        'type': 'llm_generated',
                        'insight': f"Analysis completed for {len(posts)} social media posts with {sentiment_analysis.get('overall_sentiment', 'mixed')} sentiment",
                        'confidence': 0.6,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'fallback_summary'
                    })
        
        except Exception as e:
            self.logger.warning(f"Could not generate tool-enhanced insights: {str(e)}")
            # Provide fallback insights based on available data
            if sentiment_analysis.get('overall_sentiment'):
                insights.append({
                    'type': 'sentiment_based',
                    'insight': f"Social sentiment towards fintech is {sentiment_analysis['overall_sentiment']} with score {sentiment_analysis.get('sentiment_score', 0):.2f}",
                    'confidence': 0.7,
                    'timestamp': datetime.now().isoformat()
                })
            
            if trending_topics:
                top_topic = trending_topics[0]
                insights.append({
                    'type': 'trend_based', 
                    'insight': f"'{top_topic['topic']}' is trending with {top_topic['mention_count']} mentions in social discussions",
                    'confidence': 0.8,
                    'timestamp': datetime.now().isoformat()
                })
        
        return insights
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score for fintech content using config weights."""
        score = 0.0
        text_lower = text.lower()
        
        # Use keyword weights from config
        for keyword, weight in Config.FINTECH_KEYWORD_WEIGHTS.items():
            if keyword in text_lower:
                score += weight
        
        return min(score, 1.0)
    
    def _calculate_data_quality(self, posts: List[Dict]) -> float:
        """Calculate overall data quality score using config values."""
        if not posts:
            return 0.0
        
        quality_factors = []
        
        # Text length quality using config values
        avg_length = sum(len(post['text']) for post in posts) / len(posts)
        length_score = 1.0 if Config.MIN_TEXT_LENGTH <= avg_length <= Config.MAX_TEXT_LENGTH else 0.5
        quality_factors.append(length_score)
        
        # Language quality
        english_posts = sum(1 for post in posts if self._is_english(post['text']))
        language_score = english_posts / len(posts)
        quality_factors.append(language_score)
        
        # Relevance quality
        avg_relevance = sum(post.get('relevance_score', 0) for post in posts) / len(posts)
        quality_factors.append(avg_relevance)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _is_english(self, text: str) -> bool:
        """Simple English language detection using config indicators."""
        text_lower = text.lower()
        matches = sum(1 for word in Config.ENGLISH_INDICATORS if word in text_lower)
        return matches >= Config.MIN_ENGLISH_INDICATORS