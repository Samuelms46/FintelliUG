"""
FintelliUG - Main Entry Point
Uganda Fintech Intelligence Platform

Usage:
    python main.py --app          # Run Streamlit dashboard
    python main.py --workflow     # Run agent workflow
    python main.py --setup        # Setup database and initialize
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from database.db_manager import DatabaseManager
from database.vector_db import ChromaDBManager
from utils.logger import setup_logger
from agents.langgraph_workflow import MultiAgentWorkflow
import subprocess

def setup_environment():
    """Setup environment and create necessary directories"""
    logger = setup_logger("main")
    
    # Create necessary directories
    directories = [
        "data",
        "data/chroma_db", 
        "logs",
        "data_collection/cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Check for .env file
    if not Path(".env").exists():
        logger.info("Required environment variables:")
        logger.info("- OPENAI_API_KEY")
        logger.info("- GROQ_API_KEY") 
        logger.info("- AZURE_OPENAI_API_KEY")
        logger.info("- AZURE_EMBEDDING_ENDPOINT")
        logger.info("- AZURE_EMBEDDING_BASE")
        logger.info("- REDDIT_CLIENT_ID")
        logger.info("- REDDIT_CLIENT_SECRET")
        logger.info("- X_BEARER_TOKEN")
        return False
    
    return True

def setup_database():
    """Initialize database and vector store"""
    logger = setup_logger("main")
    
    try:
        # Initialize database
        db_manager = DatabaseManager()
        logger.info("Database initialized successfully")
        
        # Initialize vector database with error handling
        try:
            vector_db = ChromaDBManager()
            logger.info("Vector database initialized successfully")
        except Exception as chroma_error:
            logger.warning(f"ChromaDB initialization failed: {chroma_error}")
            logger.info("Continuing without vector database...")
        
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit dashboard"""
    logger = setup_logger("main")
    logger.info("Starting Streamlit dashboard...")
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        logger.info("Streamlit app stopped by user")
    except Exception as e:
        logger.error(f"Failed to start Streamlit app: {e}")

def run_agent_workflow():
    """Run the multi-agent workflow"""
    logger = setup_logger("main")
    logger.info("Starting multi-agent workflow...")
    
    try:
        workflow = MultiAgentWorkflow()
        result = workflow.run()
        logger.info("Workflow completed successfully")
        logger.info(f"Result: {result}")
    except Exception as e:
        logger.error(f"Workflow failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="FintelliUG - Uganda Fintech Intelligence Platform")
    parser.add_argument("--app", action="store_true", help="Run Streamlit dashboard")
    parser.add_argument("--workflow", action="store_true", help="Run agent workflow")
    parser.add_argument("--setup", action="store_true", help="Setup database and initialize")
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_environment():
        print("Environment setup failed. Please check the logs and create a .env file.")
        sys.exit(1)
    
    # Setup database if requested
    if args.setup:
        if setup_database():
            print("Database setup completed successfully!")
        else:
            print("Database setup failed. Check the logs for details.")
            sys.exit(1)
    
    # Run requested operation
    if args.app:
        run_streamlit_app()
    elif args.workflow:
        run_agent_workflow()
    elif args.setup:
        # Setup already done above
        pass
    else:
        # Default: show help
        parser.print_help()
        print("\nQuick start:")
        print("1. Copy env_example.txt to .env and fill in your API keys")
        print("2. Run: python main.py --setup")
        print("3. Run: python main.py --app")

if __name__ == "__main__":
    main()
