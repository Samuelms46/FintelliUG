from abc import ABC, abstractmethod
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta
import os
import json
import redis
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from utils.tools import XSearchTool

# Load environment variables from .env file
load_dotenv()

class BaseAgent(ABC):
    """Base class for FintelliUG agents, providing shared functionality for NLP, vector storage, logging, and caching.

    Designed for extension by specialized agents (e.g., SocialIntelAgent, CompetitorAnalysisAgent) to process
    fintech-related data in Uganda.
    """

    def __init__(self, name: str):
        """Initialize the agent with Groq LLM, ChromaDB, Redis, and Azure embeddings from .env.
        """
        self.name = name
        required_env_vars = [
            "GROQ_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_EMBEDDING_ENDPOINT",
            "AZURE_EMBEDDING_BASE"
        ]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL"),
            temperature=float(os.getenv("GROQ_TEMPERATURE", 0.7)),
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_EMBEDDING_BASE")
        )
        self.vector_store = Chroma(
            collection_name=f"fintelliug_{name.lower().replace(' ', '_')}_data",
            embedding_function=self.embeddings,
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        )
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        self.logger = logging.getLogger(f"agent.{name}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize the XSearchTool for social data collection
        try:
            self.x_search_tool = XSearchTool()
            self.logger.info("XSearchTool initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize XSearchTool: {str(e)}")
            self.x_search_tool = None

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return insights using a functional-style pipeline.

        Args:
            input_data (Dict[str, Any]): Input data with 'text', 'source', 'timestamp'.

        Returns:
            Dict[str, Any]: Processed insights (e.g., {'insights': [...], 'error': None}).
        """
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data format for required fields.

        Args:
            input_data (Dict[str, Any]): Input data to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        required_fields = ['text', 'source', 'timestamp']
        is_valid = all(field in input_data for field in required_fields)
        if not is_valid:
            self.logger.error(f"Invalid input data: missing fields {required_fields}")
        return is_valid

    def store_in_vector_db(self, text: str, metadata: Dict[str, Any], doc_id: str):
        """Store text and metadata in ChromaDB for similarity search.

        Args:
            text (str): Text to embed and store.
            metadata (Dict[str, Any]): Metadata (e.g., source, timestamp).
            doc_id (str): Unique identifier for the document.
        """
        try:
            self.vector_store.add_texts(texts=[text], metadatas=[metadata], ids=[doc_id])
            self.logger.info(f"Stored document {doc_id} in ChromaDB")
        except Exception as e:
            self.logger.error(f"Failed to store in ChromaDB: {str(e)}")

    def query_vector_db(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query ChromaDB for similar documents.

        Args:
            query (str): Query text for similarity search.
            k (int): Number of results to return (default: 5).

        Returns:
            List[Dict[str, Any]]: List of matching documents with text, metadata, and scores.
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [{"text": doc.page_content, "metadata": doc.metadata, "score": score} for doc, score in results]
        except Exception as e:
            self.logger.error(f"ChromaDB query failed: {str(e)}")
            return []

    def cache_result(self, key: str, result: Dict[str, Any], ttl: int = 3600):
        """Cache processing result in Redis with a time-to-live (TTL).

        Args:
            key (str): Cache key (e.g., hash of input text).
            result (Dict[str, Any]): Result to cache.
            ttl (int): Time-to-live in seconds (default: 1 hour).
        """
        try:
            self.redis_client.setex(key, ttl, json.dumps(result))
            self.logger.debug(f"Cached result for key {key}")
        except Exception as e:
            self.logger.error(f"Failed to cache result: {str(e)}")

    def get_cached_result(self, key: str) -> Dict[str, Any] | None:
        """Retrieve cached result from Redis.

        Args:
            key (str): Cache key.

        Returns:
            Dict[str, Any] | None: Cached result or None if not found.
        """
        try:
            cached = self.redis_client.get(key)
            if cached:
                self.logger.debug(f"Retrieved cached result for key {key}")
                return json.loads(cached)
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve cache: {str(e)}")
            return None

    def log_performance(self, start_time: datetime, result: Dict[str, Any]):
        """Log agent performance metrics.

        Args:
            start_time (datetime): Start time of processing.
            result (Dict[str, Any]): Processing result.
        """
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        insights_count = len(result.get('insights', []))
        self.logger.info(f"{self.name} processed data in {processing_time:.2f}s")
        self.logger.info(f"Generated {insights_count} insights")

    def fetch_social_data(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Fetch social media data using the XSearchTool and return anonymized posts.

        Args:
            query (str): Search query for social media posts (e.g., fintech news)
            max_results (int): Maximum number of posts to return (default: 10).

        Returns:
            List[Dict[str, Any]]: List of anonymized posts with text, source, and timestamp.
            Returns empty list if tool is not available or on error.
        """
        if not self.x_search_tool:
            self.logger.warning("XSearchTool not available - cannot fetch social data")
            return []
        
        try:
            self.logger.info(f"Fetching social data for query: {query}")
            posts = self.x_search_tool.run(query, max_results)
            self.logger.info(f"Successfully fetched {len(posts)} social media posts")
            return posts
        except Exception as e:
            self.logger.error(f"Failed to fetch social data: {str(e)}")
            return []

    def delete_old_records(self, cutoff_days: int = 90) -> bool:
        """Delete ChromaDB records older than specified days (default: 90 days) for compliance with Uganda's Data Protection Act."""
        try:
            cutoff = (datetime.now() - timedelta(days=cutoff_days)).isoformat()
            total_before = self.vector_store._collection.count()
            self.vector_store.delete(where={"timestamp": {"$lt": cutoff}})
            total_after = self.vector_store._collection.count()
            deleted_count = total_before - total_after
            self.logger.info(f"Deleted {deleted_count} records older than {cutoff_days} days")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete old records: {str(e)}")
            return False