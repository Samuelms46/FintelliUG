from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import json

# Createing a Base class for models
Base = declarative_base()

class SocialMediaPost(Base):
    __tablename__ = 'social_media_posts'
    
    id = Column(Integer, primary_key=True)
    source = Column(String(50))
    content = Column(Text)
    cleaned_content = Column(Text)
    language = Column(String(20))
    sentiment = Column(String(20))
    sentiment_score = Column(Float)
    topics = Column(JSON)
    relevance_score = Column(Float)
    author = Column(String(100))
    url = Column(String(500))
    timestamp = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Post {self.id} from {self.source}>"

class CompetitorMention(Base):
    __tablename__ = 'competitor_mentions'
    
    id = Column(Integer, primary_key=True)
    post_id = Column(Integer)
    competitor = Column(String(100))
    sentiment = Column(String(20))
    context = Column(Text)
    extracted_insights = Column(JSON)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Insight(Base):
    __tablename__ = 'insights'
    
    id = Column(Integer, primary_key=True)
    type = Column(String(50))
    content = Column(Text)
    confidence = Column(Float)
    source_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# Exporting the Base so it can be imported by other modules
__all__ = ['Base', 'SocialMediaPost', 'CompetitorMention', 'Insight']