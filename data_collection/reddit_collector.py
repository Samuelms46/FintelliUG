import praw
import time
from datetime import datetime
from config import Config
from utils.logger import app_logger

class RedditDataCollector:
    def __init__(self):
        self.client = None
        if Config.REDDIT_CLIENT_ID and Config.REDDIT_CLIENT_SECRET:
            try:
                self.client = praw.Reddit(
                    client_id=Config.REDDIT_CLIENT_ID,
                    client_secret=Config.REDDIT_CLIENT_SECRET,
                    user_agent="fintelliug/1.0.0 (by /u/fintelliug)"
                )
                app_logger.info("Reddit client initialized successfully")
            except Exception as e:
                app_logger.error(f"Failed to initialize Reddit client: {e}")
                self.client = None
        else:
            app_logger.warning("Reddit credentials not provided. Using mock data only.")
    
    def search_uganda_fintech(self, query="Uganda fintech", limit=20, subreddits=None):
        """Search for Uganda fintech related posts"""
        if not self.client:
            return self._get_mock_data()
        
        if subreddits is None:
            subreddits = ["Uganda", "fintech", "Africa", "mobilemoney"]
        
        try:
            results = []
            search_query = f"{query} subreddit:{' OR '.join(subreddits)}"
            
            for submission in self.client.subreddit("all").search(
                search_query, 
                limit=limit, 
                time_filter="week"  # last week
            ):
                if self._is_relevant(submission):
                    results.append(self._format_post(submission))
            
            app_logger.info(f"Collected {len(results)} posts from Reddit")
            return results
            
        except Exception as e:
            app_logger.error(f"Error searching Reddit: {e}")
            return self._get_mock_data()
    
    def get_subreddit_posts(self, subreddit="Uganda", limit=25, time_filter="day"):
        """Get recent posts from a specific subreddit"""
        if not self.client:
            return self._get_mock_data(subreddit)
        
        try:
            results = []
            sub = self.client.subreddit(subreddit)
            
            for submission in sub.new(limit=limit):
                if self._is_relevant(submission):
                    results.append(self._format_post(submission))
            
            app_logger.info(f"Collected {len(results)} posts from r/{subreddit}")
            return results
            
        except Exception as e:
            app_logger.error(f"Error getting posts from r/{subreddit}: {e}")
            return self._get_mock_data(subreddit)
    
    def _is_relevant(self, submission):
        """Check if post is relevant to Uganda fintech"""
        text = (submission.title + ' ' + (submission.selftext or '')).lower()
        
        # Uganda keywords
        uganda_keywords = ['uganda', 'kampala', 'entebbe', 'ugandan', 'ugx']
        
        # Fintech keywords
        fintech_keywords = [
            'mobile money', 'mtn', 'airtel', 'bank', 'fintech', 'loan',
            'money', 'payment', 'digital wallet', 'send money', 'cash',
            'savings', 'investment', 'insurance', 'transfer', 'mobile banking'
        ]
        
        has_uganda = any(kw in text for kw in uganda_keywords)
        has_fintech = any(kw in text for kw in fintech_keywords)
        
        return has_uganda and has_fintech
    
    def _format_post(self, submission):
        """Format Reddit submission for our database"""
        return {
            'source': 'reddit',
            'content': submission.title + ' ' + (submission.selftext or ''),
            'author': str(submission.author),
            'url': f"https://reddit.com{submission.permalink}",
            'timestamp': datetime.fromtimestamp(submission.created_utc),
            'upvotes': submission.score,
            'comments': submission.num_comments,
            'subreddit': submission.subreddit.display_name
        }
    
    def _get_mock_data(self, subreddit=None):
        """Fallback mock data when Reddit API is not available"""
        mock_posts = [
            {
                'source': 'reddit',
                'content': 'MTN Mobile Money just increased fees for transactions above 100k UGX. Not happy about this!',
                'author': 'UgandaFinanceGuy',
                'url': '#',
                'timestamp': datetime.now(),
                'upvotes': 15,
                'comments': 8,
                'subreddit': subreddit or 'Uganda'
            },
            {
                'source': 'reddit', 
                'content': 'Airtel Money has better network coverage in rural areas compared to MTN. Great for sending money to village.',
                'author': 'RuralBanking',
                'url': '#',
                'timestamp': datetime.now(),
                'upvotes': 23,
                'comments': 12,
                'subreddit': subreddit or 'fintech'
            }
        ]
        return mock_posts