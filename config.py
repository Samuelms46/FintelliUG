import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fintelliug.db")
    
    # Vector Database 
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "chroma_db/chroma.sqlite3")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "fintelliug_embeddings")
    
    # Data directory
    DATA_DIR = os.getenv("DATA_DIR", "chroma_db")
    
    # Model settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") 
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    # Embedding mode
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Fintech topics for Uganda
    FINTECH_TOPICS = {
        "Mobile Money": ["mtn", "airtel money", "mobile money", "momo", "send money", "cash out"],
        "Digital Banking": ["bank", "account", "digital banking", "online banking", "agent banking"],
        "Mobile Lending": ["loan", "okash", "branch", "credit", "borrow", "lending"],
        "Savings & Investment": ["save", "investment", "interest", "savings", "invest"],
        "Cross-border Payments": ["remittance", "diaspora", "international", "send abroad", "worldremit"],
        "Insurance Technology": ["insurance", "insure", "premium", "claim", "health insurance"],
        "Regulations": ["regulation", "bank of uganda", "compliance", "license", "policy"]
    }
    
    # Competitors to track
    COMPETITORS = [
        "MTN Mobile Money",
        "Airtel Money", 
        "Chipper Cash",
        "FlexPay",
        "CentCorp",
        "Ecobank",
        "Stanbic Bank",
        "Absa Bank"
    ]