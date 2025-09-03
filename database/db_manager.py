from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base, SocialMediaPost, CompetitorMention, Insight
from .vector_db import ChromaDBManager
from config import Config
import json
from typing import Dict, List

class DatabaseManager:
    def __init__(self):
        # Enable pool_pre_ping to avoid stale connections-tune pool size to reduce QueuePool overflows
        self.engine = create_engine(
            Config.DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30
        )
        self.Session = sessionmaker(bind=self.engine)
        self.vector_db = ChromaDBManager()
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        return self.Session()
    
    def add_post(self, post_data):
        session = self.get_session()
        try:
            post = SocialMediaPost(**post_data)
            session.add(post)
            session.commit()
            return post.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_unprocessed_posts(self, limit=100):
        session = self.get_session()
        try:
            return session.query(SocialMediaPost).filter(
                SocialMediaPost.processed == False
            ).limit(limit).all()
        finally:
            session.close()
    
    def mark_post_processed(self, post_id):
        session = self.get_session()
        try:
            post = session.query(SocialMediaPost).get(post_id)
            if post:
                post.processed = True
                session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def update_post_topics(self, post_id, topics, confidence):
        session = self.get_session()
        try:
            post = session.query(SocialMediaPost).get(post_id)
            if post:
                post.topics = topics
                post.relevance_score = confidence
                session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def update_post_sentiment(self, post_id, sentiment, score):
        session = self.get_session()
        try:
            post = session.query(SocialMediaPost).get(post_id)
            if post:
                post.sentiment = sentiment
                post.sentiment_score = score
                session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def add_competitor_mention(self, mention_data):
        session = self.get_session()
        try:
            mention = CompetitorMention(**mention_data)
            session.add(mention)
            session.commit()
            return mention.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def add_insight(self, insight_data):
        session = self.get_session()
        try:
            insight = Insight(**insight_data)
            session.add(insight)
            session.commit()
            return insight.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_recent_posts(self, hours=24, limit=100):
        session = self.get_session()
        try:
            from datetime import datetime, timedelta
            time_threshold = datetime.utcnow() - timedelta(hours=hours)
            return session.query(SocialMediaPost).filter(
                SocialMediaPost.created_at >= time_threshold
            ).limit(limit).all()
        finally:
            session.close()
    
    def get_competitor_mentions(self, hours=24, competitor=None):
        session = self.get_session()
        try:
            from datetime import datetime, timedelta
            time_threshold = datetime.utcnow() - timedelta(hours=hours)
            
            query = session.query(CompetitorMention).filter(
                CompetitorMention.created_at >= time_threshold
            )
            
            if competitor:
                query = query.filter(CompetitorMention.competitor == competitor)
                
            return query.all()
        finally:
            session.close()

# Methods for vector database integration
    def add_to_vector_db(self, post_data: Dict) -> None:
        """Add a post to the vector database"""
        self.vector_db.add_documents([post_data])
    
    def search_similar_posts(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar posts in vector database"""
        return self.vector_db.search_similar(query, n_results)
    
    def search_posts_by_topic(self, topic: str, n_results: int = 10) -> List[Dict]:
        """Search posts by topic"""
        return self.vector_db.search_by_topic(topic, n_results)
    
    def search_posts_by_sentiment(self, sentiment: str, n_results: int = 10) -> List[Dict]:
        """Search posts by sentiment"""
        return self.vector_db.search_by_sentiment(sentiment, n_results)
    
    def get_vector_db_stats(self) -> Dict:
        """Get vector database statistics"""
        return self.vector_db.get_collection_stats()

    def count_posts(self):
        """Count total posts in database"""
        session = self.get_session()
        try:
            return session.query(SocialMediaPost).count()
        finally:
            session.close()

    def count_relevant_posts(self, min_relevance=0.3):
        """Count posts with minimum relevance score"""
        session = self.get_session()
        try:
           return session.query(SocialMediaPost).filter(
            SocialMediaPost.relevance_score >= min_relevance
          ).count()
        finally:
            session.close()        