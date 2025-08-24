import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
from config import Config
from utils.logger import app_logger

class ChromaDBManager:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(
                path=Config.VECTOR_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.collection = self._get_or_create_collection()
            app_logger.info("ChromaDB manager initialized successfully")
        except Exception as e:
            app_logger.error(f"Failed to initialize ChromaDB: {e}")
            # Fallback to mock implementation
            self.client = None
            self.embedding_model = None
            self.collection = None
    
    def _get_or_create_collection(self):
        """Get existing collection or create a new one"""
        try:
            collection = self.client.get_collection(Config.COLLECTION_NAME)
            app_logger.info(f"Using existing collection: {Config.COLLECTION_NAME}")
            return collection
        except:
            app_logger.info(f"Creating new collection: {Config.COLLECTION_NAME}")
            return self.client.create_collection(
                name=Config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity
            )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if self.embedding_model is None:
            # Return dummy embeddings if model not available
            return [[0.1] * 384 for _ in texts]  # 384-dim dummy embeddings
        
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector database"""
        if not documents or self.client is None:
            app_logger.warning("Cannot add documents - ChromaDB not initialized")
            return
        
        try:
            # Extract data for Chroma
            ids = [str(doc.get("id", i)) for i, doc in enumerate(documents)]
            texts = [doc.get("content", "") for doc in documents]
            metadatas = [
                {
                    "source": doc.get("source", "unknown"),
                    "post_id": doc.get("post_id", ""),
                    "topics": str(doc.get("topics", [])),
                    "sentiment": doc.get("sentiment", "neutral"),
                    "timestamp": str(doc.get("timestamp", "")),
                    "author": doc.get("author", "unknown")
                }
                for doc in documents
            ]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            app_logger.info(f"Added {len(documents)} documents to ChromaDB")
            
        except Exception as e:
            app_logger.error(f"Error adding documents to ChromaDB: {e}")
    
    def search_similar(self, query: str, n_results: int = 5, 
                      filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents"""
        if self.client is None:
            app_logger.warning("ChromaDB not initialized - returning empty results")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            app_logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def search_by_topic(self, topic: str, n_results: int = 10) -> List[Dict]:
        """Search for documents related to a specific topic"""
        return self.search_similar(
            query=topic,
            n_results=n_results,
            filters={"topics": {"$contains": topic}} if topic else None
        )
    
    def search_by_sentiment(self, sentiment: str, n_results: int = 10) -> List[Dict]:
        """Search for documents with specific sentiment"""
        return self.search_similar(
            query="",  # Empty query to use filter only
            n_results=n_results,
            filters={"sentiment": sentiment} if sentiment else None
        )
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        if self.client is None:
            return {
                "total_documents": 0,
                "collection_name": "not_initialized",
                "status": "error"
            }
        
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": Config.COLLECTION_NAME,
                "status": "active"
            }
        except Exception as e:
            app_logger.error(f"Error getting collection stats: {e}")
            return {
                "total_documents": 0,
                "collection_name": Config.COLLECTION_NAME,
                "status": "error"
            }
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the collection"""
        if self.client is None:
            return
        
        try:
            self.collection.delete(ids=ids)
            app_logger.info(f"Deleted {len(ids)} documents from ChromaDB")
        except Exception as e:
            app_logger.error(f"Error deleting documents: {e}")