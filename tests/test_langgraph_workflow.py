import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.langgraph_workflow import MultiAgentWorkflow
from agents.state import AgentState

class TestLangGraphWorkflow:
    def setup_method(self):
        self.workflow = MultiAgentWorkflow()
        
    def test_workflow_structure(self):
        """Test that workflow has correct structure"""
        assert hasattr(self.workflow, 'workflow')
        assert self.workflow.workflow is not None
        
        # Verify all required nodes are present
        required_nodes = ["fetch_data", "process_posts", "analyze_competitors", "generate_insights", "compile_report"]
        for node in required_nodes:
            assert node in self.workflow.workflow.nodes, f"Workflow should contain {node} node"
    
    def test_initial_state(self):
        """Test that initial state has correct structure"""
        initial_state = AgentState(
            messages=[],
            raw_posts=[],
            processed_posts=[],
            competitor_mentions=[],
            market_insights=[],
            final_report=None,
            errors=[]
        )
        
        # Verify state structure
        assert isinstance(initial_state["messages"], list)
        assert isinstance(initial_state["raw_posts"], list)
        assert isinstance(initial_state["processed_posts"], list)
        assert isinstance(initial_state["competitor_mentions"], list)
        assert isinstance(initial_state["market_insights"], list)
        assert initial_state["final_report"] is None
        assert isinstance(initial_state["errors"], list)
    
    def test_individual_nodes(self):
        """Test each node individually with mock data"""
        # Test fetch_data node
        state = AgentState(
            messages=[],
            raw_posts=[],
            processed_posts=[],
            competitor_mentions=[],
            market_insights=[],
            final_report=None,
            errors=[]
        )
        
        # Mock the database call
        import unittest.mock
        with unittest.mock.patch.object(self.workflow.db_manager, 'get_unprocessed_posts') as mock_get:
            mock_get.return_value = []
            result = self.workflow.fetch_data(state)
            assert "raw_posts" in result
            assert isinstance(result["raw_posts"], list)