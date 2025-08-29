import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.competitor_agent import CompetitorAnalysisAgent


class TestCompetitorAnalysisAgent:
    """Comprehensive unit tests for CompetitorAnalysisAgent class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Set up environment variable patches first
        self.env_patcher = patch.dict(os.environ, {
            'GROQ_API_KEY': 'test_key',
            'AZURE_OPENAI_API_KEY': 'test_key', 
            'AZURE_EMBEDDING_ENDPOINT': 'test_endpoint',
            'AZURE_EMBEDDING_BASE': 'test_base',
            'GROQ_MODEL': 'test_model',
            'GROQ_TEMPERATURE': '0.7'
        })
        self.env_patcher.start()
        
        # Mock all the dependencies
        self.groq_patcher = patch('agents.base_agent.ChatGroq')
        self.embeddings_patcher = patch('agents.base_agent.AzureOpenAIEmbeddings')
        self.chroma_patcher = patch('agents.base_agent.Chroma')
        self.redis_patcher = patch('agents.base_agent.redis.Redis')
        self.xsearch_patcher = patch('agents.base_agent.XSearchTool')
        
        # Start all patchers
        self.mock_groq = self.groq_patcher.start()
        self.mock_embeddings = self.embeddings_patcher.start()
        self.mock_chroma = self.chroma_patcher.start()
        self.mock_redis = self.redis_patcher.start()
        self.mock_xsearch = self.xsearch_patcher.start()
        
        # Create the agent
        self.agent = CompetitorAnalysisAgent()
        
        # Mock the dependencies
        self.agent.llm = Mock()
        self.agent.vector_store = Mock()
        self.agent.redis_client = Mock()
        self.agent.logger = Mock()
    
    def teardown_method(self):
        """Clean up patches after each test method."""
        self.env_patcher.stop()
        self.groq_patcher.stop()
        self.embeddings_patcher.stop()
        self.chroma_patcher.stop()
        self.redis_patcher.stop()
        self.xsearch_patcher.stop()
    
    def test_init_default_config(self):
        """Test initialization with default configuration."""
        with patch.dict(os.environ, {
            'GROQ_API_KEY': 'test_key',
            'AZURE_OPENAI_API_KEY': 'test_key',
            'AZURE_EMBEDDING_ENDPOINT': 'test_endpoint',
            'AZURE_EMBEDDING_BASE': 'test_base'
        }):
            with patch('agents.base_agent.ChatGroq'), \
                 patch('agents.base_agent.AzureOpenAIEmbeddings'), \
                 patch('agents.base_agent.Chroma'), \
                 patch('agents.base_agent.redis.Redis'), \
                 patch('agents.base_agent.XSearchTool'):
                
                agent = CompetitorAnalysisAgent()
                
                assert agent.name == "competitor_analysis"
                assert len(agent.competitors) == 7
                assert "MTN MoMo" in agent.competitors
                assert "Airtel Money" in agent.competitors
                assert "Chipper Cash" in agent.competitors
    
    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            "competitors": ["MTN MoMo", "Custom Bank"]
        }
        
        with patch.dict(os.environ, {
            'GROQ_API_KEY': 'test_key',
            'AZURE_OPENAI_API_KEY': 'test_key',
            'AZURE_EMBEDDING_ENDPOINT': 'test_endpoint',
            'AZURE_EMBEDDING_BASE': 'test_base'
        }):
            with patch('agents.base_agent.ChatGroq'), \
                 patch('agents.base_agent.AzureOpenAIEmbeddings'), \
                 patch('agents.base_agent.Chroma'), \
                 patch('agents.base_agent.redis.Redis'), \
                 patch('agents.base_agent.XSearchTool'):
                
                agent = CompetitorAnalysisAgent(custom_config)
                
                assert len(agent.competitors) == 2
                assert "Custom Bank" in agent.competitors
    
    def test_validate_input_valid_query(self):
        """Test input validation with valid query data."""
        input_data = {"query": "MTN MoMo"}
        assert self.agent.validate_input(input_data) is True
    
    def test_validate_input_valid_competitor(self):
        """Test input validation with valid competitor data."""
        input_data = {"competitor": "Airtel Money"}
        assert self.agent.validate_input(input_data) is True
    
    def test_validate_input_valid_posts(self):
        """Test input validation with valid posts data."""
        input_data = {"posts": [{"text": "Test post"}]}
        assert self.agent.validate_input(input_data) is True
    
    def test_validate_input_valid_hours(self):
        """Test input validation with valid hours data."""
        input_data = {"hours": 24}
        assert self.agent.validate_input(input_data) is True
    
    def test_validate_input_invalid_non_dict(self):
        """Test input validation with non-dictionary input."""
        input_data = "not a dict"
        assert self.agent.validate_input(input_data) is False
        self.agent.logger.error.assert_called_with("Input must be a dictionary")
    
    def test_validate_input_invalid_empty(self):
        """Test input validation with empty dictionary."""
        input_data = {}
        assert self.agent.validate_input(input_data) is False
    
    def test_validate_input_invalid_posts_not_list(self):
        """Test input validation with posts that is not a list."""
        input_data = {"posts": "not a list"}
        assert self.agent.validate_input(input_data) is False
    
    def test_create_cache_key_competitor_and_hours(self):
        """Test cache key creation with competitor and hours."""
        input_data = {"competitor": "MTN MoMo", "hours": 48}
        expected = "competitor_analysis_comp_MTN MoMo_hours_48"
        assert self.agent._create_cache_key(input_data) == expected
    
    def test_create_cache_key_posts_only(self):
        """Test cache key creation with posts only."""
        input_data = {"posts": [{"text": "post1"}, {"text": "post2"}]}
        expected = "competitor_analysis_posts_2"
        assert self.agent._create_cache_key(input_data) == expected
    
    def test_create_cache_key_basic(self):
        """Test cache key creation with minimal input."""
        input_data = {}
        expected = "competitor_analysis"
        assert self.agent._create_cache_key(input_data) == expected
    
    @patch('agents.competitor_agent.datetime')
    def test_analyze_posts(self, mock_datetime):
        """Test posts analysis functionality."""
        mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T00:00:00"
        
        test_posts = [
            {"text": "MTN MoMo is great for payments"},
            {"text": "Airtel Money has better rates"}
        ]
        
        # Mock the competitor insights extraction
        self.agent._extract_competitor_insights_from_post = Mock(side_effect=[
            [{"competitor": "MTN MoMo", "sentiment": "positive"}],
            [{"competitor": "Airtel Money", "sentiment": "positive"}]
        ])
        
        # Mock summary generation
        self.agent._generate_summary_insights = Mock(return_value=[
            {"competitor": "MTN MoMo", "overall_sentiment": "positive"}
        ])
        
        result = self.agent._analyze_posts(test_posts)
        
        assert result["analysis_type"] == "posts"
        assert result["total_posts_analyzed"] == 2
        assert len(result["competitor_mentions"]) == 2
        assert "summary" in result
        assert "timestamp" in result
    
    def test_extract_competitor_insights_from_post(self):
        """Test competitor insights extraction from single post."""
        content = "I love using MTN MoMo for mobile payments. It's very reliable."
        
        # Mock sentiment analysis
        self.agent._analyze_competitor_sentiment = Mock(return_value={
            "sentiment": "positive",
            "key_points": ["reliable service"],
            "confidence": 0.8
        })
        
        with patch('agents.competitor_agent.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T00:00:00"
            
            insights = self.agent._extract_competitor_insights_from_post(1, content)
        
        assert len(insights) == 1
        assert insights[0]["competitor"] == "MTN MoMo"
        assert insights[0]["sentiment"] == "positive"
        assert insights[0]["post_id"] == 1
        assert insights[0]["confidence"] == 0.8
    
    def test_extract_competitor_insights_multiple_competitors(self):
        """Test insights extraction with multiple competitors in one post."""
        content = "MTN MoMo and Airtel Money are both good options"
        
        self.agent._analyze_competitor_sentiment = Mock(return_value={
            "sentiment": "positive",
            "key_points": ["good service"],
            "confidence": 0.7
        })
        
        with patch('agents.competitor_agent.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T00:00:00"
            
            insights = self.agent._extract_competitor_insights_from_post(1, content)
        
        assert len(insights) == 2
        competitors = [insight["competitor"] for insight in insights]
        assert "MTN MoMo" in competitors
        assert "Airtel Money" in competitors
    
    def test_extract_competitor_insights_no_competitors(self):
        """Test insights extraction with no competitor mentions."""
        content = "This is just a regular post about payments"
        
        insights = self.agent._extract_competitor_insights_from_post(1, content)
        
        assert len(insights) == 0
    
    def test_analyze_competitor_sentiment_success(self):
        """Test successful competitor sentiment analysis."""
        content = "MTN MoMo service is excellent and fast"
        competitor = "MTN MoMo"
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '''
        {
            "sentiment": "positive",
            "key_points": ["excellent service", "fast transactions"],
            "confidence": 0.9,
            "competitive_aspect": "service"
        }
        '''
        self.agent.llm.invoke.return_value = mock_response
        
        result = self.agent._analyze_competitor_sentiment(content, competitor)
        
        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.9
        assert "excellent service" in result["key_points"]
    
    def test_analyze_competitor_sentiment_fallback(self):
        """Test competitor sentiment analysis fallback on error."""
        content = "MTN MoMo service"
        competitor = "MTN MoMo"
        
        # Mock LLM to raise exception
        self.agent.llm.invoke.side_effect = Exception("LLM error")
        
        result = self.agent._analyze_competitor_sentiment(content, competitor)
        
        assert result["sentiment"] == "neutral"
        assert result["confidence"] == 0.3
        assert result["competitive_aspect"] == "general"
        assert "Mention of MTN MoMo detected" in result["key_points"]
    
    def test_analyze_competitor_sentiment_invalid_json(self):
        """Test sentiment analysis with invalid JSON response."""
        content = "MTN MoMo service"
        competitor = "MTN MoMo"
        
        # Mock LLM response with invalid JSON
        mock_response = Mock()
        mock_response.content = "This is not valid JSON"
        self.agent.llm.invoke.return_value = mock_response
        
        result = self.agent._analyze_competitor_sentiment(content, competitor)
        
        # Should fall back to default analysis
        assert result["sentiment"] == "neutral"
        assert result["confidence"] == 0.3
    
    def test_fallback_analysis(self):
        """Test fallback analysis method."""
        competitor = "Test Bank"
        
        result = self.agent._fallback_analysis(competitor)
        
        assert result["sentiment"] == "neutral"
        assert result["confidence"] == 0.3
        assert result["competitive_aspect"] == "general"
        assert f"Mention of {competitor} detected" in result["key_points"]
    
    def test_generate_summary_insights_empty_list(self):
        """Test summary insights generation with empty list."""
        result = self.agent._generate_summary_insights([])
        assert result == []
    
    def test_generate_summary_insights_single_competitor(self):
        """Test summary insights generation for single competitor."""
        mentions = [
            {
                "competitor": "MTN MoMo",
                "sentiment": "positive"
            },
            {
                "competitor": "MTN MoMo", 
                "sentiment": "negative"
            },
            {
                "competitor": "MTN MoMo",
                "sentiment": "positive"
            }
        ]
        
        result = self.agent._generate_summary_insights(mentions)
        
        assert len(result) == 1
        summary = result[0]
        assert summary["competitor"] == "MTN MoMo"
        assert summary["overall_sentiment"] == "positive"
        assert summary["total_mentions"] == 3
        assert summary["confidence"] > 0.5
    
    def test_generate_summary_insights_multiple_competitors(self):
        """Test summary insights generation for multiple competitors."""
        mentions = [
            {"competitor": "MTN MoMo", "sentiment": "positive"},
            {"competitor": "Airtel Money", "sentiment": "negative"},
            {"competitor": "MTN MoMo", "sentiment": "positive"}
        ]
        
        result = self.agent._generate_summary_insights(mentions)
        
        assert len(result) == 2
        competitors = [s["competitor"] for s in result]
        assert "MTN MoMo" in competitors
        assert "Airtel Money" in competitors
    
    def test_generate_summary_insights_error_handling(self):
        """Test summary insights generation with error handling."""
        # Malformed mention data to trigger exception
        mentions = [{"invalid": "data"}]
        
        result = self.agent._generate_summary_insights(mentions)
        
        assert result == []
        self.agent.logger.error.assert_called()
    
    def test_generate_competitive_landscape(self):
        """Test competitive landscape generation."""
        competitor_summaries = {
            "MTN MoMo": {
                "total_mentions": 10,
                "summary": [{"overall_sentiment": "positive"}]
            },
            "Airtel Money": {
                "total_mentions": 5,
                "summary": [{"overall_sentiment": "negative"}]
            }
        }
        
        result = self.agent._generate_competitive_landscape(competitor_summaries)
        
        assert "market_leaders" in result
        assert "sentiment_leaders" in result
        assert "mention_volume" in result
        assert "key_insights" in result
        assert result["mention_volume"]["MTN MoMo"] == 10
        assert result["mention_volume"]["Airtel Money"] == 5
    
    def test_process_invalid_input(self):
        """Test process method with invalid input."""
        input_data = "invalid"
        
        result = self.agent.process(input_data)
        
        assert "error" in result
        assert result["error"] == "Invalid input data"
    
    def test_process_with_cache_hit(self):
        """Test process method with cache hit."""
        input_data = {"competitor": "MTN MoMo"}
        cached_result = {"cached": True}
        
        self.agent.get_cached_result = Mock(return_value=cached_result)
        
        result = self.agent.process(input_data)
        
        assert result == cached_result
        self.agent.logger.info.assert_called_with("Returning cached competitor analysis")
    
    @patch('agents.competitor_agent.datetime')
    def test_process_posts_analysis(self, mock_datetime):
        """Test process method with posts analysis."""
        mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T00:00:00"
        
        input_data = {
            "posts": [{"text": "MTN MoMo is great"}]
        }
        
        self.agent.get_cached_result = Mock(return_value=None)
        self.agent.cache_result = Mock()
        self.agent._analyze_posts = Mock(return_value={"analysis": "posts"})
        
        result = self.agent.process(input_data)
        
        assert result == {"analysis": "posts"}
        self.agent._analyze_posts.assert_called_once_with([{"text": "MTN MoMo is great"}])
        self.agent.cache_result.assert_called_once()
    
    def test_process_competitor_analysis(self):
        """Test process method with specific competitor analysis."""
        input_data = {
            "competitor": "MTN MoMo",
            "hours": 48
        }
        
        self.agent.get_cached_result = Mock(return_value=None)
        self.agent.cache_result = Mock()
        self.agent._analyze_competitor = Mock(return_value={"analysis": "competitor"})
        
        result = self.agent.process(input_data)
        
        assert result == {"analysis": "competitor"}
        self.agent._analyze_competitor.assert_called_once_with("MTN MoMo", 48)
    
    def test_process_general_intelligence(self):
        """Test process method with general competitive intelligence."""
        input_data = {"hours": 72}
        
        self.agent.get_cached_result = Mock(return_value=None) 
        self.agent.cache_result = Mock()
        self.agent._generate_competitive_intelligence = Mock(return_value={"analysis": "general"})
        
        result = self.agent.process(input_data)
        
        assert result == {"analysis": "general"}
        self.agent._generate_competitive_intelligence.assert_called_once_with(72)
    
    def test_process_exception_handling(self):
        """Test process method exception handling."""
        input_data = {"competitor": "MTN MoMo"}
        
        self.agent.get_cached_result = Mock(return_value=None)
        self.agent._analyze_competitor = Mock(side_effect=Exception("Test error"))
        
        result = self.agent.process(input_data)
        
        assert "error" in result
        assert "Analysis failed: Test error" in result["error"]
        self.agent.logger.error.assert_called()
    
    def test_analyze_competitor_with_vector_db(self):
        """Test competitor analysis using vector database."""
        competitor = "MTN MoMo"
        hours = 24
        
        # Mock vector database query results
        mock_results = [
            {
                "text": "MTN MoMo service is excellent",
                "metadata": {
                    "post_id": "post_1",
                    "timestamp": datetime.now().isoformat()
                },
                "score": 0.9
            }
        ]
        
        self.agent.query_vector_db = Mock(return_value=mock_results)
        self.agent._is_recent_post = Mock(return_value=True)
        self.agent._analyze_competitor_sentiment = Mock(return_value={
            "sentiment": "positive",
            "key_points": ["excellent service"],
            "confidence": 0.8
        })
        self.agent._store_competitor_analysis = Mock()
        self.agent._generate_summary_insights = Mock(return_value=[])
        
        with patch('agents.competitor_agent.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T00:00:00"
            
            result = self.agent._analyze_competitor(competitor, hours)
        
        assert result["analysis_type"] == "competitor_specific"
        assert result["competitor"] == competitor
        assert result["time_period_hours"] == hours
        assert result["total_mentions"] == 1
        
        self.agent.query_vector_db.assert_called_once_with(competitor, k=50)
        self.agent._store_competitor_analysis.assert_called_once()
    
    def test_generate_competitive_intelligence(self):
        """Test comprehensive competitive intelligence generation."""
        # Mock the _analyze_competitor method for each competitor
        self.agent._analyze_competitor = Mock(return_value={
            "mentions": [{"competitor": "MTN MoMo", "sentiment": "positive"}],
            "total_mentions": 1
        })
        
        self.agent._generate_summary_insights = Mock(return_value=[])
        self.agent._generate_competitive_landscape = Mock(return_value={})
        
        with patch('agents.competitor_agent.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T00:00:00"
            
            result = self.agent._generate_competitive_intelligence(48)
        
        assert result["analysis_type"] == "comprehensive"
        assert result["time_period_hours"] == 48
        assert result["competitors_analyzed"] == len(self.agent.competitors)
        assert "competitor_summaries" in result
        assert "overall_summary" in result
        assert "competitive_landscape" in result
        
        # Should call _analyze_competitor for each competitor
        assert self.agent._analyze_competitor.call_count == len(self.agent.competitors)


if __name__ == "__main__":
    pytest.main([__file__])