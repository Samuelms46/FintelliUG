# FintelliUG Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites
- Python 3.8 or higher
- API keys for OpenAI, Groq, Azure OpenAI, Reddit, and Twitter/X

### Step 1: Clone and Setup
```bash
# Navigate to your project directory
cd FintelliUG

# Install dependencies
python setup.py

# Or manually install requirements
pip install -r requirements.txt
```

### Step 2: Configure Environment
```bash
# Copy the example environment file
cp env_template.txt .env

# Edit .env with your API keys
# Required keys:
# - OPENAI_API_KEY
# - GROQ_API_KEY  
# - AZURE_OPENAI_API_KEY
# - AZURE_EMBEDDING_ENDPOINT
# - AZURE_EMBEDDING_BASE
# - REDDIT_CLIENT_ID
# - REDDIT_CLIENT_SECRET
# - X_BEARER_TOKEN
```

### Step 3: Initialize Database
```bash
# Setup database and vector store
python main.py --setup
```

### Step 4: Run the Application
```bash
# Start the Streamlit dashboard
python main.py --app

# Or run the agent workflow directly
python main.py --workflow
```

### Step 5: Access the Dashboard
Open your browser and go to: `http://localhost:8501`

## 🔧 Troubleshooting

### Common Issues

**1. Missing API Keys**
```
Error: Missing required environment variables
```
**Solution**: Make sure all required API keys are set in your `.env` file.

**2. Database Connection Issues**
```
Error: Database setup failed
```
**Solution**: Check that SQLite can write to the project directory.

**3. Vector Database Issues**
```
Error: ChromaDB initialization failed
```
**Solution**: Ensure the `data/chroma_db` directory exists and is writable.

**4. Import Errors**
```
ModuleNotFoundError: No module named 'langchain'
```
**Solution**: Run `pip install -r requirements.txt` to install all dependencies.

### Getting API Keys

**OpenAI API Key**
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add to `.env` as `OPENAI_API_KEY=your_key_here`

**Groq API Key**
1. Go to https://console.groq.com/
2. Create an account and get your API key
3. Add to `.env` as `GROQ_API_KEY=your_key_here`

**Azure OpenAI**
1. Set up Azure OpenAI service
2. Get your endpoint and API key
3. Add to `.env`:
   ```
   AZURE_OPENAI_API_KEY=your_key_here
   AZURE_EMBEDDING_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_EMBEDDING_BASE=your-deployment-name
   ```

**Reddit API**
1. Go to https://www.reddit.com/prefs/apps
2. Create a new app
3. Get Client ID and Client Secret
4. Add to `.env`:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   ```

**Twitter/X API**
1. Go to https://developer.twitter.com/
2. Create a new app
3. Get your Bearer Token
4. Add to `.env` as `X_BEARER_TOKEN=your_token_here`

## 📊 Using the Dashboard

### Dashboard Features
1. **Data Collection**: Collect data from Reddit and other sources
2. **Social Posts**: View and analyze social media posts
3. **Competitor Analysis**: Track competitor mentions and sentiment
4. **Vector Search**: Search for similar content using AI
5. **Insights**: View AI-generated market insights

### Running the Workflow
1. Click "Collect & Process New Data" to gather fresh data
2. Click "Run Intelligence Workflow" to analyze the data
3. View results in the various dashboard tabs

## 🔄 Development Workflow

### Making Changes
1. Edit your code
2. Test with `python main.py --workflow`
3. Run the dashboard with `python main.py --app`

### Adding New Agents
1. Create a new agent class inheriting from `BaseAgent`
2. Implement the `process()` method
3. Add to the workflow in `agents/langgraph_workflow.py`

### Adding New Data Sources
1. Create a new collector in `data_collection/`
2. Update the data processing pipeline
3. Add to the dashboard if needed

## 📞 Support

If you encounter issues:
1. Check the logs in the `logs/` directory
2. Ensure all API keys are correctly set
3. Verify all dependencies are installed
4. Check the troubleshooting section above

For additional help, contact:
- Email: rogerskalema0@gmail.com
- Phone: +256-751612501
