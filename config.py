import os
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for FintelliUG application."""

    # ==================== API KEYS ====================
    # OpenAI/Azure OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
    AZURE_EMBEDDING_BASE = os.getenv("AZURE_EMBEDDING_BASE")

    # Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.7"))

    # Social Media APIs
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

    # ==================== DATABASE CONFIG ====================
    # SQL Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fintelliug.db")

    # Vector Database
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "chroma_db")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

    # Redis Cache
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

    # ==================== VECTOR DB COLLECTIONS ====================
    DEFAULT_COLLECTION_NAME = os.getenv("DEFAULT_COLLECTION_NAME", "fintelliug_default")

    AGENT_COLLECTIONS: Dict[str, str] = {
        "competitor_analysis": "fintelliug_competitors",
        "sentiment_analysis": "fintelliug_sentiment",
        "trend_analysis": "fintelliug_trends",
        "market_analysis": "fintelliug_market",
        "social_intelligence": "fintelliug_social"
    }

    # ==================== MODEL SETTINGS ====================
    AZURE_EMBEDDING_MODEL = "text-embedding-3-small"

    # ==================== SHARED AGENT SETTINGS ====================
    # These apply to all agents unless overridden
    DEFAULT_CACHE_TTL = int(os.getenv("DEFAULT_CACHE_TTL", "3600"))  # 1 hour
    DEFAULT_MAX_POSTS = int(os.getenv("DEFAULT_MAX_POSTS", "10"))
    DEFAULT_RELEVANCE_THRESHOLD = float(os.getenv("DEFAULT_RELEVANCE_THRESHOLD", "0.45"))

    # Data Quality Settings (shared across agents)
    MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "50"))
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "500"))
    MIN_ENGLISH_INDICATORS = int(os.getenv("MIN_ENGLISH_INDICATORS", "3"))

    # English language indicators for detection
    ENGLISH_INDICATORS = [
        'the', 'and', 'is', 'to', 'of', 'in', 'for', 'on', 'with',
        'at', 'by', 'this', 'that', 'are', 'was', 'be', 'have', 'has'
    ]

    # Keyword weights for relevance scoring
    FINTECH_KEYWORD_WEIGHTS = {
        'mobile money': 0.3,
        'fintech': 0.2,
        'digital payments': 0.2,
        'mtn momo': 0.25,
        'mtn mobile money': 0.25,
        'airtel money': 0.25,
        'chipper cash': 0.25,
        'banking': 0.15,
        'uganda': 0.15,
        'payment': 0.1,
        'transfer': 0.1,
        'financial': 0.1,
        'wallet': 0.1,
        'loan': 0.15,
        'savings': 0.15,
        'investment': 0.15
    }

    # ==================== AGENT-SPECIFIC OVERRIDES ====================
    AGENT_CONFIGS = {
        "social_intelligence": {
            "cache_ttl": int(os.getenv("SOCIAL_INTEL_CACHE_TTL", "1800")), 
            "max_posts": int(os.getenv("SOCIAL_INTEL_MAX_POSTS", "20")),   
            "relevance_threshold": float(os.getenv("SOCIAL_INTEL_RELEVANCE_THRESHOLD", "0.45")),
            "trending_days": int(os.getenv("SOCIAL_INTEL_TRENDING_DAYS", 7)),
            "trending_threshold": float(os.getenv("SOCIAL_INTEL_TRENDING_THRESHOLD", "0.1"))
        },
        "competitor_analysis": {
            "cache_ttl": int(os.getenv("COMPETITOR_CACHE_TTL", "7200")), 
            "max_posts": int(os.getenv("COMPETITOR_MAX_POSTS", "100")),
            "relevance_threshold": float(os.getenv("COMPETITOR_RELEVANCE_THRESHOLD", "0.35")),
            "analysis_depth": os.getenv("COMPETITOR_ANALYSIS_DEPTH", "detailed")
        },
        "sentiment_analysis": {
            "cache_ttl": int(os.getenv("SENTIMENT_CACHE_TTL", "1800")),
            "max_posts": int(os.getenv("SENTIMENT_MAX_POSTS", "50")),
            "sentiment_model": os.getenv("SENTIMENT_MODEL", "vader")
        }
    }

    # ==================== BUSINESS LOGIC CONFIG ====================
    # Fintech topics for Uganda
    FINTECH_TOPICS: Dict[str, List[str]] = {
        "Mobile Money": [
            "mtn mobile money", "airtel money", "mobile money", "momo",
            "send money", "cash out", "mobile wallet"
        ],
        "Digital Banking": [
            "digital banking", "online banking", "agent banking",
            "bank account", "mobile banking", "internet banking"
        ],
        "Mobile Lending": [
            "okash", "branch app", "credit", "loan", "borrow",
            "lending", "quick loan", "instant loan"
        ],
        "Savings & Investment": [
            "savings", "investment", "interest rate", "fixed deposit",
            "mutual funds", "bonds", "stocks"
        ],
        "Cross-border Payments": [
            "remittance", "diaspora money", "international transfer",
            "send abroad", "worldremit", "western union", "moneygram"
        ],
        "Insurance Technology": [
            "insurance", "health insurance", "life insurance", "premium",
            "claim", "micro insurance", "crop insurance"
        ],
        "Regulations": [
            "bank of uganda", "bou", "regulation", "compliance",
            "license", "fintech policy", "kyc", "aml"
        ],
        "Cryptocurrency": [
            "bitcoin", "cryptocurrency", "crypto", "blockchain",
            "digital currency", "binance", "web3"
        ]
    }

    # Major competitors in Uganda's fintech space
    COMPETITORS: List[str] = [
        # Mobile Money Leaders
        "MTN Mobile Money",
        "Airtel Money",

        # Digital Payment Platforms
        "Chipper Cash",
        "FlexPay",
        "Flutterwave",
        "Payway",
        "Pesapal",
        "Eversend",
        "Wave",

        # Traditional Banks (Digital Services)
        "Stanbic Bank",
        "Centenary Bank",
        "DFCU Bank",
        "Equity Bank",
        "Standard Chartered",

        # Fintech Startups
        "Ezee Money",
        "Mcash",
        "Interswitch",
        "Pegasus Technologies",

        # Multi-service Platforms
        "Safeboda",
        "Uber",
        "Bolt"
    ]

    # ==================== APPLICATION SETTINGS ====================
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Cache TTL (Time To Live) in seconds
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default

    # Data retention (days)
    DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "90"))

    # Social media fetch limits
    MAX_SOCIAL_POSTS = int(os.getenv("MAX_SOCIAL_POSTS", "100"))

    # Analysis timeframes
    DEFAULT_ANALYSIS_HOURS = int(os.getenv("DEFAULT_ANALYSIS_HOURS", "24"))

    @classmethod
    def get_agent_config(cls, agent_type: str, key: str, default=None):
        """Get agent-specific config with fallback to default."""
        agent_configs = cls.AGENT_CONFIGS.get(agent_type, {})
        if key in agent_configs:
            return agent_configs[key]
        
        # Fallback to default config
        default_key = f"DEFAULT_{key.upper()}"
        return getattr(cls, default_key, default)

    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate required configuration values."""
        missing = []

        # Required API keys
        required_keys = [
            "GROQ_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_EMBEDDING_ENDPOINT",
            "AZURE_EMBEDDING_BASE"
        ]

        for key in required_keys:
            if not getattr(cls, key):
                missing.append(key)

        return missing

    @classmethod
    def get_competitor_keywords(cls, competitor: str) -> List[str]:
        """Get search keywords for a specific competitor."""
        # Generate variations of competitor names for better matching
        base_name = competitor.lower()
        keywords = [base_name, competitor]

        # Add common variations
        if "mobile money" in base_name:
            keywords.extend([base_name.replace("mobile money", "momo")])
        if "bank" in base_name:
            keywords.extend([base_name.replace("bank", "")])

        return list(set(keywords))  # Remove duplicates

    @classmethod
    def get_fintech_keywords(cls) -> List[str]:
        """Get all fintech-related keywords for general searches."""
        all_keywords = []
        for topic_keywords in cls.FINTECH_TOPICS.values():
            all_keywords.extend(topic_keywords)
        return list(set(all_keywords))  # Remove duplicates

    @classmethod
    def get_all_fintech_keywords(cls) -> List[str]:
        """Get flattened list of all fintech keywords including competitors."""
        keywords = []
        for topic_keywords in cls.FINTECH_TOPICS.values():
            keywords.extend(topic_keywords)
        # Add competitor names as keywords
        keywords.extend([comp.lower() for comp in cls.COMPETITORS])
        return list(set(keywords))  # Remove duplicates