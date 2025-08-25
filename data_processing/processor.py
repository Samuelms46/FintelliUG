from .cleaner import DataCleaner
from .topic_extractor import TopicExtractor
from database.db_manager import DatabaseManager
from config import Config
from data_collection.reddit_collector import RedditDataCollector

class DataProcessor:
    def __init__(self):
        self.cleaner = DataCleaner()
        self.topic_extractor = TopicExtractor()
        self.db_manager = DatabaseManager()
        self.reddit_collector = RedditDataCollector()
    
    def collect_reddit_data(self, query="Uganda fintech", limit=20):
        """Collect data from Reddit API"""
        posts = self.reddit_collector.search_uganda_fintech(query, limit)
        return self.batch_process(posts)

    def process_post(self, raw_post):
        """Process a single social media post"""
        # Clean the text
        cleaned_content = self.cleaner.clean_text(raw_post.get("content", ""))
        
        # Skip if not relevant
        if not self.cleaner.is_relevant(cleaned_content):
            return None
        
        # Detect language
        language = self.cleaner.detect_language(cleaned_content)
        
        # Calculate relevance score
        relevance_score = self.cleaner.calculate_relevance_score(cleaned_content)
        
        # Prepare post data for database
        post_data = {
            "source": raw_post.get("source", "unknown"),
            "content": raw_post.get("content", ""),
            "cleaned_content": cleaned_content,
            "language": language,
            "relevance_score": relevance_score,
            "author": raw_post.get("author", "unknown"),
            "url": raw_post.get("url", ""),
            "timestamp": raw_post.get("timestamp")
        }
        
        # Add to database
        post_id = self.db_manager.add_post(post_data)
        
        # Extract topics
        topics, confidence = self.topic_extractor.extract_topics(cleaned_content)
        
        # Update post with topics
        self.db_manager.update_post_topics(post_id, topics, confidence)
        
        # ADD THIS: Add to vector database
        vector_data = {
            "id": str(post_id),
            "content": cleaned_content,
            "source": raw_post.get("source", "unknown"),
            "post_id": post_id,
            "topics": topics,
            "sentiment": "neutral",  # Will be updated later
            "timestamp": str(raw_post.get("timestamp")),
            "author": raw_post.get("author", "unknown")
        }
        self.db_manager.add_to_vector_db(vector_data)
        
        return {
            "post_id": post_id,
            "topics": topics,
            "confidence": confidence
        }
    
    def batch_process(self, raw_posts):
        """Process multiple posts in batch"""
        results = []
        for post in raw_posts:
            try:
                result = self.process_post(post)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing post: {e}")
        return results