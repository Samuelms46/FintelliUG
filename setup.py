"""
FintelliUG Setup Script
Installs dependencies and sets up the project environment
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    directories = [
        "logs",
        "data_collection/cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def check_env_file():
    """Check if .env file exists"""
    if not Path(".env").exists():
        print("⚠️  No .env file found!")
        print("Required environment variables:")
        print("- OPENAI_API_KEY")
        print("- GROQ_API_KEY")
        print("- AZURE_OPENAI_API_KEY") 
        print("- AZURE_EMBEDDING_ENDPOINT")
        print("- AZURE_EMBEDDING_BASE")
        print("- REDDIT_CLIENT_ID")
        print("- REDDIT_CLIENT_SECRET")
        print("- X_BEARER_TOKEN")
        return False
    else:
        print("✅ .env file found")
        return True

def main():
    print("🚀 FintelliUG Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check environment file
    env_ok = check_env_file()
    
    print("\n" + "=" * 50)
    if env_ok:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python main.py --setup")
        print("2. Run: python main.py --app")
    else:
        print("⚠️  Setup completed with warnings!")
        print("Please create a .env file before running the application.")

if __name__ == "__main__":
    main()
