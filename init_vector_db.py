
"""
Initialize the Chroma vector database with sample data
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from database.vector_db import ChromaDBManager
from database.db_manager import DatabaseManager
from config import Config

def init_vector_db():
    """Initialize the vector database with sample data"""
    print("Initializing Chroma vector database...")
    
    vector_db = ChromaDBManager()
    db_manager = DatabaseManager()
    
    # Sample data for testing
    sample_documents = [
        {
            "id": "sample_1",
            "content": "MTN Mobile Money fees are too high for small transactions",
            "source": "sample",
            "post_id": 1001,
            "topics": ["Mobile Money", "Fees"],
            "sentiment": "negative",
            "timestamp": "2023-11-01T10:00:00",
            "author": "sample_user"
        },
        {
            "id": "sample_2", 
            "content": "Airtel Money has better network coverage in rural areas of Uganda",
            "source": "sample",
            "post_id": 1002,
            "topics": ["Mobile Money", "Coverage"],
            "sentiment": "positive", 
            "timestamp": "2023-11-01T11:30:00",
            "author": "sample_user"
        },
        {
            "id": "sample_3",
            "content": "Bank of Uganda announces new regulations for fintech companies",
            "source": "sample", 
            "post_id": 1003,
            "topics": ["Regulations", "Digital Banking"],
            "sentiment": "neutral",
            "timestamp": "2023-11-02T09:15:00",
            "author": "sample_user"
        }
    ]
    
    # Add sample documents
    vector_db.add_documents(sample_documents)
    
    # Test the search functionality
    results = vector_db.search_similar("mobile money fees", n_results=2)
    print(f"Found {len(results)} similar documents")
    
    # Show collection stats
    stats = vector_db.get_collection_stats()
    print(f"Vector DB initialized with {stats['total_documents']} documents")
    
    print("Chroma vector database initialization complete!")

if __name__ == "__main__":
    init_vector_db()