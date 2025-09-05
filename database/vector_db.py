import chromadb
from chromadb.config import Settings
from langchain_openai import AzureOpenAIEmbeddings
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil
import os
import time

from config import Config
from utils.logger import app_logger


class ChromaDBManager:
    """Manager for ChromaDB vector database operations with multi-collection support."""

    _instances: Dict[str, "ChromaDBManager"] = {}
    _client: Optional[Any] = None

    def __new__(cls, collection_name: Optional[str] = None) -> "ChromaDBManager":
        collection_name = collection_name or Config.DEFAULT_COLLECTION_NAME
        
        if collection_name not in cls._instances:
            cls._instances[collection_name] = super(ChromaDBManager, cls).__new__(cls)
            cls._instances[collection_name]._initialized = False
            
        return cls._instances[collection_name]

    def __init__(self, collection_name: Optional[str] = None) -> None:
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.collection_name = collection_name or Config.DEFAULT_COLLECTION_NAME
        self.collection: Optional[Any] = None
        # use Azure embedder from langchain_openai
        self.embedder: Optional[AzureOpenAIEmbeddings] = None

        try:
            self._initialize_database()
            self._initialized = True
            app_logger.info(f"ChromaDB manager initialized for collection: {self.collection_name}")
        except Exception as e:
            app_logger.error(f"Failed to initialize ChromaDB for {self.collection_name}: {e}")
            self._initialized = True

    def _initialize_database(self) -> None:
        """Initialize the ChromaDB client and embedding model."""
        db_path = Path(Config.VECTOR_DB_PATH)
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize shared client if not exists
        if ChromaDBManager._client is None:
            ChromaDBManager._client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(anonymized_telemetry=False),
            )
        
        self.client = ChromaDBManager._client
        # Initialize Azure embeddings
        try:
            model_name = getattr(Config, "AZURE_EMBEDDING_MODEL", None)
            if not model_name:
                raise RuntimeError("Config.AZURE_EMBEDDING_MODEL not set")
            azure_endpoint = getattr(Config, "AZURE_EMBEDDING_ENDPOINT", None) or os.getenv("AZURE_EMBEDDING_ENDPOINT")
            if not azure_endpoint:
                raise RuntimeError("Azure endpoint not configured. Set AZURE_EMBEDDING_ENDPOINT in your config or environment.")
            self.embedder = AzureOpenAIEmbeddings(model=model_name, azure_endpoint=azure_endpoint)
            app_logger.info(f"Initialized Azure embeddings with model: {model_name} and endpoint: {azure_endpoint}")
        except Exception as e:
            app_logger.error(f"Failed to initialize Azure embeddings: {e}")
            self.embedder = None
        self.collection = self._get_or_create_collection()
        app_logger.info(f"Initializing ChromaDB at path: {db_path} for collection: {self.collection_name}")

    def _get_or_create_collection(self) -> Optional[chromadb.Collection]:
        """Get existing collection or create a new one."""
        if not self.client:
            return None

        try:
            collection = self.client.get_collection(self.collection_name)
            app_logger.info(f"Using existing collection: {self.collection_name}")
            return collection
        except Exception:
            app_logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        # Use Azure embedder if available
        if self.embedder is not None:
            try:
                # embed_documents returns List[List[float]]
                return self.embedder.embed_documents(texts)
            except Exception as e:
                app_logger.error(f"Azure embedding call failed: {e}")

        # Fallback: return dummy zero embeddings with common dim (384)
        dim = getattr(Config, "EMBEDDING_DIM", 384)
        app_logger.warning("Returning dummy embeddings; configure Config.AZURE_EMBEDDING_MODEL and Azure creds.")
        return [[0.0] * dim for _ in texts]

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector database"""
        if not documents or self.client is None or self.collection is None:
            app_logger.warning("Cannot add documents - ChromaDB not initialized")
            return

        try:
            # Extract data for Chroma
            ids = [str(doc.get("id", i)) for i, doc in enumerate(documents)]
            texts = [doc.get("content", "") for doc in documents]
            metadatas = [
                {
                    "source": str(doc.get("source", "unknown")),
                    "post_id": str(doc.get("post_id", "")),
                    "topics": str(doc.get("topics", [])),
                    "sentiment": str(doc.get("sentiment", "neutral")),
                    "timestamp": str(doc.get("timestamp", "")),
                    "author": str(doc.get("author", "unknown")),
                }
                for doc in documents
            ]

            # Generate embeddings
            embeddings = self.generate_embeddings(texts)

            # Add to collection
            self.collection.add(
                ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts
            )

            app_logger.info(f"Added {len(documents)} documents to ChromaDB")

        except Exception as e:
            app_logger.error(f"Error adding documents to ChromaDB: {e}")

    def search_similar(
        self, query: str, n_results: int = 5, filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar documents"""
        if self.client is None or self.collection is None:
            app_logger.warning("ChromaDB not initialized - returning empty results")
            return []

        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]

            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=n_results, where=filters
            )

            # Format results
            formatted_results = []
            if results and "ids" in results and results["ids"]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append(
                        {
                            "id": results["ids"][0][i],
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i],
                        }
                    )

            return formatted_results

        except Exception as e:
            app_logger.error(f"Error searching ChromaDB: {e}")
            return []

    def search_by_topic(self, topic: str, n_results: int = 10) -> List[Dict]:
        """Search for documents related to a specific topic"""
        return self.search_similar(
            query=topic,
            n_results=n_results,
            filters={"topics": {"$contains": topic}} if topic else None,
        )

    def search_by_sentiment(self, sentiment: str, n_results: int = 10) -> List[Dict]:
        """Search for documents with specific sentiment"""
        return self.search_similar(
            query="",  # Empty query to use filter only
            n_results=n_results,
            filters={"sentiment": sentiment} if sentiment else None,
        )

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        if self.client is None or self.collection is None:
            return {
                "total_documents": 0,
                "collection_name": self.collection_name,
                "status": "error",
            }

        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "status": "active",
            }
        except Exception as e:
            app_logger.error(f"Error getting collection stats: {e}")
            return {
                "total_documents": 0,
                "collection_name": self.collection_name,
                "status": "error",
            }

    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the collection"""
        if self.client is None or self.collection is None:
            app_logger.warning("ChromaDB not initialized - cannot delete documents")
            return

        try:
            self.collection.delete(ids=ids)
            app_logger.info(f"Deleted {len(ids)} documents from ChromaDB")
        except Exception as e:
            app_logger.error(f"Error deleting documents: {e}")

    def reset_database(self) -> bool:
        """Reset the ChromaDB database by recreating it."""
        try:
            # Set initialized to False to force reinitialization
            self._initialized = False

            # Get the directory path
            db_path = Path(Config.VECTOR_DB_PATH)

            # If it's a file path, get its parent
            if db_path.suffix:
                db_path = db_path.parent

            # If the directory exists, rename it as backup
            if db_path.exists():
                backup_path = db_path.with_name(
                    f"{db_path.name}_backup_{int(time.time())}"
                )
                shutil.move(str(db_path), str(backup_path))

            # Create a fresh directory
            os.makedirs(str(db_path), exist_ok=True)

            # Reinitialize
            self.__init__()
            return True
        except Exception as e:
            app_logger.error(f"Failed to reset ChromaDB: {e}")
            return False
