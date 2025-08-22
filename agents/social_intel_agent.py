from .base_agent import BaseAgent
from typing import Dict, List, Any
import json
from datetime import datetime, timedelta

class SocialIntelAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("social_intelligence")
        self.fintech_keywords = [
            'mobile money', 'fintech', 'digital payments', 'MTN MoMo', 
            'Airtel Money', 'banking', 'loans', 'savings', 'investment'
        ]
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        # Check for 'posts' key and that it's a non-empty list of dicts with 'text' and 'timestamp'
        if 'posts' not in input_data or not isinstance(input_data['posts'], list) or not input_data['posts']:
            if hasattr(self, 'logger'):
                self.logger.error("Input data must contain a non-empty 'posts' list.")
            return False
        for post in input_data['posts']:
            if not isinstance(post, dict) or 'text' not in post or 'timestamp' not in post:
                if hasattr(self, 'logger'):
                    self.logger.error("Each post must be a dict with 'text' and 'timestamp'.")
                return False
        return True

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now()
        
        if not self.validate_input(input_data):
            return {'error': 'Invalid input data'}
        
        # Extract fintech-related content
        relevant_posts = self._filter_fintech_content(input_data['posts'])
        
        # Analyze sentiment trends
        sentiment_analysis = self._analyze_sentiment_trends(relevant_posts)
        
        # Detect emerging topics
        trending_topics = self._detect_trending_topics(relevant_posts)
        
        # Generate insights
        insights = self._generate_social_insights(
            sentiment_analysis, trending_topics
        )
        
        result = {
            'agent': self.name,
            'timestamp': datetime.now().isoformat(),
            'sentiment_analysis': sentiment_analysis,
            'trending_topics': trending_topics,
            'insights': insights,
            'data_quality_score': self._calculate_data_quality(relevant_posts)
        }
        
        self.log_performance(start_time, result)
        return result
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        # Check for 'posts' key and that it's a non-empty list of dicts with 'text' and 'timestamp'
        if 'posts' not in input_data or not isinstance(input_data['posts'], list) or not input_data['posts']:
            self.logger.error("Input data must contain a non-empty 'posts' list.")
            return False
        for post in input_data['posts']:
            if not isinstance(post, dict) or 'text' not in post or 'timestamp' not in post:
                self.logger.error("Each post must be a dict with 'text' and 'timestamp'.")
                return False
        return True
    
    def _filter_fintech_content(self, posts: List[Dict]) -> List[Dict]:
        """Filter posts for fintech-related content."""
        relevant_posts = []
        
        for post in posts:
            text = post.get('text', '').lower()
            if any(keyword in text for keyword in self.fintech_keywords):
                post['relevance_score'] = self._calculate_relevance(text)
                if post['relevance_score'] > 0.6:
                    relevant_posts.append(post)
        
        return relevant_posts
    
    def _analyze_sentiment_trends(self, posts: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment trends over time."""
        prompt = f"""
        Analyze sentiment trends in these Uganda fintech discussions:

        Posts: {json.dumps([p['topic'] for p in posts[:10]])}

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
            post_time = datetime.fromisoformat(post['timestamp'])
            if (current_time - post_time).days <= 7:  # Last 7 days
                for keyword in self.fintech_keywords:
                    if keyword in post['text'].lower():
                        topic_counts[keyword] = topic_counts.get(keyword, 0) + 1
        
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
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score for fintech content."""
        score = 0.0
        text_lower = text.lower()
        
        # Keyword matching with weights
        keyword_weights = {
            'mobile money': 0.3,
            'fintech': 0.2,
            'digital payments': 0.2,
            'banking': 0.1,
            'uganda': 0.2
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