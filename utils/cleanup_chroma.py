"""
Utility to clean up ChromaDB conflicts and reset vector database
"""

import os
import shutil
from pathlib import Path
import logging

def setup_logger():
    """Setup logging for cleanup operations"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def cleanup_chroma_db():
    """Clean up ChromaDB directory to resolve conflicts"""
    logger = setup_logger()
    
    # ChromaDB paths to check
    chroma_paths = [
        "./data/chroma_db",
        "./chroma_db", 
        "./data/chroma_db/chroma.sqlite3"
    ]
    
    logger.info("🔍 Checking for ChromaDB conflicts...")
    
    for path in chroma_paths:
        if os.path.exists(path):
            logger.warning(f"Found existing ChromaDB path: {path}")
            
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    logger.info(f"Removed file: {path}")
                else:
                    shutil.rmtree(path)
                    logger.info(f"Removed directory: {path}")
            except Exception as e:
                logger.error(f"Failed to remove {path}: {e}")
    
    # Create fresh directory
    try:
        os.makedirs("./data/chroma_db", exist_ok=True)
        logger.info("✅ Created fresh ChromaDB directory: ./data/chroma_db")
    except Exception as e:
        logger.error(f"Failed to create directory: {e}")

def reset_vector_database():
    """Reset the entire vector database"""
    logger = setup_logger()
    
    logger.info("🔄 Resetting vector database...")
    
    # Clean up ChromaDB
    cleanup_chroma_db()
    
    # Also clean up any SQLite database files
    sqlite_files = [
        "./fintelliug.db",
        "./data/fintelliug.db"
    ]
    
    for db_file in sqlite_files:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
                logger.info(f"Removed SQLite database: {db_file}")
            except Exception as e:
                logger.error(f"Failed to remove {db_file}: {e}")
    
    logger.info("✅ Vector database reset complete")

def check_chroma_status():
    """Check the current status of ChromaDB"""
    logger = setup_logger()
    
    logger.info("📊 Checking ChromaDB status...")
    
    chroma_dir = "./data/chroma_db"
    
    if os.path.exists(chroma_dir):
        files = os.listdir(chroma_dir)
        logger.info(f"ChromaDB directory exists with {len(files)} files")
        for file in files:
            logger.info(f"  - {file}")
    else:
        logger.info("ChromaDB directory does not exist")

def main():
    """Main cleanup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ChromaDB Cleanup Utility")
    parser.add_argument("--cleanup", action="store_true", help="Clean up ChromaDB conflicts")
    parser.add_argument("--reset", action="store_true", help="Reset entire vector database")
    parser.add_argument("--status", action="store_true", help="Check ChromaDB status")
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_chroma_db()
    elif args.reset:
        reset_vector_database()
    elif args.status:
        check_chroma_status()
    else:
        parser.print_help()
        print("\nUsage examples:")
        print("  python utils/cleanup_chroma.py --cleanup  # Clean up conflicts")
        print("  python utils/cleanup_chroma.py --reset    # Reset everything")
        print("  python utils/cleanup_chroma.py --status   # Check status")

if __name__ == "__main__":
    main()
