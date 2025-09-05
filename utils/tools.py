from langchain_core.tools import Tool
from typing import Dict, List, Any
import os
import tweepy
import logging
from dotenv import load_dotenv
import tweepy
from datetime import datetime
from utils.compliance import anonymize_text
from utils.logger import setup_logger

# Load environment variables
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class XSearchTool:
    """Custom tool for searching X posts using v2 API - simplified without LangChain Tool inheritance."""

    def __init__(self):
        self.name = "x_search"
        self.description = "Fetches recent X posts using a semantic or keyword query. " \
                          "Returns anonymized structured data: list of " \
                          "{'text': str, 'source': 'twitter', 'timestamp': iso_str}. " \
                          "Use for fintech monitoring in Uganda."
        
        # Set up logger with absolute path
        logs_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(logs_dir, exist_ok=True)  # Ensure logs directory exists
        log_file = os.path.join(logs_dir, "xsearch.log")
        self.logger = setup_logger("XSearchTool", log_file)
        
        self.bearer_token = os.getenv("X_BEARER_TOKEN")
        if not self.bearer_token:
            raise ValueError("Missing X_BEARER_TOKEN in .env")

        # Authenticate with Tweepy for v2 API
        self.client = tweepy.Client(bearer_token=self.bearer_token)

    def run(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Public interface for running the tool."""
        return self._run(query, max_results)

    def _run(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Run X search query and return anonymized posts.

        Args:
            query (str): Search query (e.g., 'fintech Uganda lang:en').
            max_results (int): Maximum posts to return (default: 10, respects free tier limits).

        Returns:
            List[Dict[str, Any]]: Anonymized posts with text, source, timestamp.
        """
        try:
            # Fetch recent tweets using v2 endpoint
            response = self.client.search_recent_tweets(
                query=query,
                tweet_fields=['created_at', 'text'],
                max_results=max_results
            )

            # use getattr to avoid Pylance attribute warnings and be defensive at runtime
            tweets_data = getattr(response, "data", None)
            if not response or not tweets_data:
                self.logger.warning(f"No tweets found for query: {query} | response: {response}")
                return []

            posts = []
            for tweet in tweets_data:
                # defensive access to fields
                text = getattr(tweet, "text", "") or ""
                try:
                    anonymized_text = anonymize_text(str(text), self.logger)
                except Exception:
                    anonymized_text = str(text)

                created_at = getattr(tweet, "created_at", None)
                ts = created_at.isoformat() if created_at else datetime.utcnow().isoformat()

                posts.append({
                    "text": anonymized_text,
                    "source": "twitter",
                    "timestamp": ts
                })

            self.logger.info(f"Fetched {len(posts)} X posts for query: {query}")
            return posts

        except tweepy.TweepyException as e:
            self.logger.error(f"X API error: {str(e)}")
            if "429" in str(e):  # Rate limit error
                self.logger.warning("Rate limit hit; retry later")
            return []

        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return []